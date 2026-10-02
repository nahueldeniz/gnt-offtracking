"""Regression tests for the path geometry and the comparison methods.

The path tests exist because a geometry error here is invisible: the simulation
runs perfectly happily against a path with corners in it, and the off-tracking it
reports is then partly a property of the corner rather than of the vehicle.
"""

import numpy as np
import pytest

from gntpf import GNT, PATHS, make_path
from gntpf.baselines import chain_condition, scale_to_bounds, steady_state_betas
from gntpf.mp_refgen import MPReferenceGenerator, closed_form_betas, joint_rhs

SMOOTH = ["agricultural", "circle", "lemniscate", "rounded_rect", "corridor"]


@pytest.mark.parametrize("name", sorted(PATHS))
def test_no_duplicate_samples(name):
    p = make_path(name)
    assert np.hypot(*np.diff(p.xy, axis=1)).min() > 1e-9


@pytest.mark.parametrize("name", SMOOTH)
def test_paths_are_g1(name):
    """No corners: the heading must not step by more than the sampling resolution."""
    p = make_path(name)
    d = np.diff(p.xy, axis=1)
    th = np.unwrap(np.arctan2(d[1], d[0]))
    assert np.rad2deg(np.abs(np.diff(th)).max()) < 1.0


def test_agricultural_curvature_is_bounded_by_turn_radius():
    p = make_path("agricultural", r=2.0)
    assert p.curvature().max() == pytest.approx(0.5, abs=5e-3)


def test_agricultural_rows_and_omega_turns():
    """Five rows of 7 m joined by four omega turns of radius 2 m at 1.5 m spacing:
    each turn is three tangent arcs, r (pi + 4 phi) long."""
    r, l, d = 2.0, 7.0, 1.5
    phi = np.arctan2(np.sqrt(4 * r**2 - (d / 2 + r) ** 2), d / 2 + r)
    p = make_path("agricultural")
    assert p.length == pytest.approx(5 * l + 4 * r * (np.pi + 4 * phi), abs=0.02)
    assert np.max(np.abs(p.curvature())) == pytest.approx(1.0 / r, rel=0.01)


@pytest.mark.parametrize("h", [(0.0, 1.0), (0.0, -1.0)])
def test_omega_turn_reverses_heading_inside_the_headland(h):
    from gntpf.paths import _omega_turn
    p1, p2, h1 = np.array([0.0, 0.0]), np.array([1.5, 0.0]), np.array(h)
    t = _omega_turn(p1, h1, p2, -h1, 2.0, ds=0.002)
    dxy = np.diff(t, axis=1)
    th = np.unwrap(np.arctan2(dxy[1], dxy[0]))
    assert abs(th[-1] - th[0]) == pytest.approx(np.pi, abs=0.01)     # a half turn, not more
    assert np.min((t - p1[:, None]).T @ h1) > -1e-9                  # never re-enters the field
    assert np.allclose(t[:, -1], p2, atol=1e-6)                       # ends on the next row


def test_signed_deviation_has_a_sign():
    p = make_path("circle", R=4.0, cx=0.0, cy=0.0)
    outside = p.signed_deviation([5.0, 0.0])
    inside = p.signed_deviation([3.0, 0.0])
    assert outside * inside < 0
    assert abs(outside) == pytest.approx(1.0, abs=1e-2)


def test_scale_to_bounds_preserves_direction():
    """Michalek's scaling keeps the commanded curvature; clipping would not."""
    u = np.array([6.0, 3.0])
    s = scale_to_bounds(u, [-2, -2], [2, 2])
    assert np.all(np.abs(s) <= 2 + 1e-12)
    assert s[0] / s[1] == pytest.approx(u[0] / u[1])


def test_chain_condition_detects_on_axle():
    assert np.isinf(chain_condition(GNT(N=2, Lh=[0.3, 0.0], L=[1.0, 0.8])))
    assert np.isfinite(chain_condition(GNT(N=2, Lh=[0.3, 0.2], L=[1.0, 0.8])))


@pytest.mark.parametrize("Lh,L", [([0.342, 0.25], [1.08, 0.78]),
                                  ([0.342, 0.0], [1.08, 0.78]),
                                  ([0.0, 0.0], [1.08, 0.78]),
                                  ([1.497] * 3, [6.303] * 3)])
@pytest.mark.parametrize("R", [3.0, 8.0, -5.0])
def test_closed_form_is_an_equilibrium(Lh, L, R):
    """Michalek & Pazderski Eqs. (32)-(37) must annihilate the joint dynamics."""
    m = GNT(N=len(Lh), Lh=Lh, L=L)
    b = closed_form_betas(m, 1.0 / R)
    d, _ = joint_rhs(m, b, np.array([1.0 / R, 1.0]), np.zeros(m.N))
    assert np.abs(d).max() < 1e-12


def test_mp_fit_recovers_the_closed_form():
    m = GNT(N=2, Lh=[0.342, 0.0], L=[1.08, 0.78])
    g = MPReferenceGenerator(m, n_harm=6, n_coll=120, homotopy=3)
    path = make_path("circle", R=6.0, cx=0.0, cy=0.0)
    s = np.linspace(0.0, path.length, 120)
    beta = g.fit(g._signed_curvature(path, s), s)
    exact = closed_form_betas(m, float(np.median(g._signed_curvature(path, s))))
    assert np.rad2deg(np.abs(beta - exact)).max() < 1.0


def test_on_axle_joint_follows_the_algebraic_relation():
    """For Lh = 0, tan(beta) = L * kappa exactly -- their Eq. (29)."""
    m = GNT(N=2, Lh=[0.342, 0.0], L=[1.08, 0.78])
    for R in (4.0, 9.0):
        b = closed_form_betas(m, 1.0 / R)
        assert b[1] == pytest.approx(np.arctan(m.L[1] / R), abs=1e-9)


def test_zero_dynamics_eigenvalues_are_v_over_lh():
    """Proposition: the linearised joint dynamics have spectrum {v / Lh_i}."""
    import casadi as ca

    Lh, L, v = [0.4, 0.2, 0.3], [1.0, 0.8, 0.6], 1.0
    m = GNT(N=3, Lh=Lh, L=L)
    b = ca.MX.sym("b", m.N)
    M = ca.MX.eye(2)
    for i in range(m.N):
        M = ca.mtimes(M, ca.vertcat(
            ca.horzcat(-L[i] * ca.cos(b[i]) / Lh[i], ca.sin(b[i]) / Lh[i]),
            ca.horzcat(L[i] * ca.sin(b[i]), ca.cos(b[i]))))
    u0 = ca.mtimes(M, ca.vertcat(0.0, v))
    chain, rows = [ca.MX.eye(2)], []
    for k in range(m.N):
        Jk = ca.vertcat(ca.horzcat(-Lh[k] * ca.cos(b[k]) / L[k], ca.sin(b[k]) / L[k]),
                        ca.horzcat(Lh[k] * ca.sin(b[k]), ca.cos(b[k])))
        chain.append(ca.mtimes(Jk, chain[k]))
    for k in range(m.N):
        Jk = ca.vertcat(ca.horzcat(-Lh[k] * ca.cos(b[k]) / L[k], ca.sin(b[k]) / L[k]),
                        ca.horzcat(Lh[k] * ca.sin(b[k]), ca.cos(b[k])))
        rows.append(ca.mtimes(ca.DM([[1.0, 0.0]]),
                              ca.mtimes(ca.MX.eye(2) - Jk, ca.mtimes(chain[k], u0))))
    A = np.asarray(ca.Function("A", [b], [ca.jacobian(ca.vertcat(*rows), b)])(np.zeros(m.N)))
    assert np.allclose(np.tril(A, -1), 0.0, atol=1e-12)          # upper triangular
    assert np.allclose(np.sort(np.diag(A)), np.sort(v / np.array(Lh)))


def test_steady_state_betas_match_the_closed_form_for_the_tractor():
    """Relaxing the chain with the tractor on radius R reproduces the SNT radii."""
    m = GNT(N=2, Lh=[0.0, 0.0], L=[1.0, 0.8])
    b = steady_state_betas(m, 1.0 / 5.0)
    R = [5.0]
    for L in m.L:
        R.append(np.sqrt(R[-1] ** 2 - L ** 2))
    assert b[0] == pytest.approx(np.arcsin(m.L[0] / R[0]), abs=1e-4)


# --------------------------------------------------------------------------- #
#  Virtual guidance point
# --------------------------------------------------------------------------- #
def test_guide_vector_normalises_and_defaults():
    from gntpf import RefGenWeights
    assert np.allclose(RefGenWeights().guide_vector(2), [0, 0, 1])
    assert np.allclose(RefGenWeights(guide_alpha=[0, 1, 3]).guide_vector(2), [0, 0.25, 0.75])
    assert RefGenWeights(guide_alpha=[1, 1, 1]).guide_vector(2).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [[1, 1], [-1, 1, 1], [0, 0, 0]])
def test_guide_vector_rejects_malformed(bad):
    from gntpf import RefGenWeights
    with pytest.raises(ValueError):
        RefGenWeights(guide_alpha=bad).guide_vector(2)


def test_guidance_point_is_held_to_the_path():
    """The objective must actually bind: the combination lands on the path.

    This is the control for the claim made in S8.  The interesting result there
    is that the *posture* can be badly wrong while this quantity is small, so it
    matters that the small value is real and not an artefact of the term being
    silently inactive.
    """
    from gntpf import RefGenWeights, SimConfig, make_path, simulate

    m = GNT(N=2, Lh=[0.342, 0.0], L=[1.08, 0.78])
    path = make_path("circle", R=4.0, cx=0.0, cy=0.0)
    alpha = np.array([0.0, 0.5, 0.5])
    res = simulate(m, path, SimConfig(sigma=1.0, t_final=14.0, seed=0),
                   refgen_weights=RefGenWeights(w_path=0.0, w_theta=5.0,
                                                w_guide=300.0, guide_alpha=alpha))
    k0 = int(0.4 * res.t.size)
    P = np.stack([res.qref[2 * m.N + 1 + 2 * i: 2 * m.N + 3 + 2 * i, k0:]
                  for i in range(m.N + 1)])
    g = np.tensordot(alpha, P, axes=(0, 0))
    dev = np.mean([path.deviation(g[:, j]) for j in range(0, g.shape[1], 20)])
    assert dev < 0.05


def test_guidance_point_off_the_tractor_leaves_the_reference_free():
    """Referencing one point anywhere but the tractor admits a drifting posture.

    The tractor is the exception because its pose plus the joint angles
    determine every other segment, so pinning it pins the chain; pinning any
    other combination does not.  The test asserts the ordering rather than a
    magnitude, which is the part that is a property of the objective.
    """
    from gntpf import RefGenWeights, SimConfig, make_path, simulate

    m = GNT(N=2, Lh=[0.342, 0.0], L=[1.08, 0.78])
    path = make_path("circle", R=4.0, cx=0.0, cy=0.0)

    def worst(alpha):
        res = simulate(m, path, SimConfig(sigma=1.0, t_final=14.0, seed=0),
                       refgen_weights=RefGenWeights(w_path=0.0, w_theta=5.0,
                                                    w_guide=300.0, guide_alpha=alpha))
        return res.summary()["worst_curved"]

    assert worst([1, 0, 0]) < 0.5 * worst([0, 0, 1])
