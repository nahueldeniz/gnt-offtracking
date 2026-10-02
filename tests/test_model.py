"""Verification of the GNT kinematics against invariants and closed-form results.

Run with ``pytest`` from the repository root.  These are the checks reported in
the paper's implementation section: they establish that the model used by the
controller, the estimator and the plant is the intended one.
"""

import numpy as np
import pytest

from gntpf import GNT

GEOMETRIES = [
    (1, [0.342], [1.08]),
    (2, [0.342, 0.0], [1.08, 0.78]),
    # the three hitching classes: Standard (every hitch on the axle),
    # non-Standard (every hitch off it) and Generalised (mixed)
    (3, [0.0, 0.0, 0.0], [1.0, 0.8, 0.7]),
    (3, [0.25, -0.2, 0.3], [1.0, 0.8, 0.7]),
    (4, [0.342, 0.0, 0.15, -0.15], [1.08, 0.78, 0.70, 0.70]),
    (7, [0.342, 0.0, 0.15, -0.15, 0.0, -0.342, 0.342],
        [0.38, 1.08, 0.70, 0.70, 0.70, 0.70, 0.70]),
]


def chain_residual(m, q):
    """Largest violation of the rigid-body geometry and of beta = theta_{i-1} - theta_i."""
    th = q[m.i_theta]
    x0, y0 = q[2 * m.N + 1], q[2 * m.N + 2]
    ax = ay = 0.0
    worst = 0.0
    for k in range(m.N):
        ax += m.Lh[k] * np.cos(th[k]) + m.L[k] * np.cos(th[k + 1])
        ay += m.Lh[k] * np.sin(th[k]) + m.L[k] * np.sin(th[k + 1])
        worst = max(worst, np.linalg.norm(q[m.i_pos(k + 1)] - np.array([x0 - ax, y0 - ay])))
    beta_err = np.max(np.abs(q[m.i_beta] - (th[:-1] - th[1:])))
    return worst, beta_err


@pytest.mark.parametrize("N,Lh,L", GEOMETRIES)
def test_chain_geometry_is_invariant(N, Lh, L):
    """The algebraic constraints of the chain are preserved by the flow."""
    rng = np.random.default_rng(1)
    m = GNT(N=N, Lh=Lh, L=L, Ts=0.01)
    th = np.cumsum(rng.uniform(-0.4, 0.4, N + 1))
    q = m.state_from_pose(*rng.uniform(-2, 2, 2), th)
    for k in range(2000):
        q = m.step(q, [0.6 * np.sin(k / 70), 1.2])
        g, b = chain_residual(m, q)
        assert g < 1e-7 and b < 1e-10


def test_snt_rotation_does_not_move_trailers():
    """With on-axle hitching, turning on the spot leaves the trailers put."""
    m = GNT(N=3, Lh=[0, 0, 0], L=[1.0, 0.8, 0.7], Ts=0.05)
    q = m.state_from_pose(1.0, 2.0, [0.1, 0.05, 0.0, -0.05])
    p0 = m.positions(q).copy()
    for _ in range(50):
        q = m.step(q, [1.5, 0.0])
    assert np.abs(m.positions(q) - p0).max() < 1e-12


def test_offaxle_hitch_drags_trailers():
    """Off-axle hitching couples tractor rotation into trailer translation --
    the effect that distinguishes a GNT from an SNT."""
    m = GNT(N=3, Lh=[0.3, -0.2, 0.0], L=[1.0, 0.8, 0.7], Ts=0.05)
    q = m.state_from_pose(1.0, 2.0, [0.1, 0.05, 0.0, -0.05])
    p0 = m.positions(q).copy()
    for _ in range(50):
        q = m.step(q, [1.5, 0.0])
    assert np.abs(m.positions(q) - p0).max() > 1e-2


def test_straight_driving_straightens_the_chain():
    m = GNT(N=3, Lh=[0, 0, 0], L=[1.0, 0.8, 0.7], Ts=0.05)
    q = m.state_from_pose(0, 0, [0.3, 0.15, 0.0, -0.15])
    norms = [np.linalg.norm(q[m.i_beta])]
    for _ in range(400):
        q = m.step(q, [0.0, 1.0])
        norms.append(np.linalg.norm(q[m.i_beta]))
    norms = np.array(norms)
    assert np.all(np.diff(norms) <= 1e-12)
    assert norms[-1] < 1e-6


def test_straight_chain_stays_straight():
    m = GNT(N=3, Lh=[0, 0, 0], L=[1.0, 0.8, 0.7], Ts=0.05)
    q = m.state_from_pose(0, 0, [0, 0, 0, 0])
    for _ in range(500):
        q = m.step(q, [0.0, 1.0])
    assert np.abs(q[m.i_beta]).max() < 1e-12


def test_steady_circle_matches_closed_form():
    """At constant curvature an SNT settles on radii R_i^2 = R_{i-1}^2 - L_i^2."""
    m = GNT(N=2, Lh=[0, 0], L=[1.0, 0.8], Ts=0.002)
    R, v = 5.0, 1.0
    q = m.state_from_pose(R, 0.0, [np.pi / 2] * 3)
    for _ in range(40000):
        q = m.step(q, [v / R, v])
    radii = np.linalg.norm(m.positions(q), axis=1)
    theory = [R]
    for L in m.L:
        theory.append(np.sqrt(theory[-1] ** 2 - L**2))
    assert np.abs(radii - np.array(theory)).max() < 1e-9


@pytest.mark.parametrize("N,Lh,L", GEOMETRIES)
def test_state_from_pose_is_consistent(N, Lh, L):
    m = GNT(N=N, Lh=Lh, L=L)
    q = m.state_from_pose(3.0, -1.0, np.linspace(0.2, -0.2, N + 1))
    g, b = chain_residual(m, q)
    assert g < 1e-12 and b < 1e-12


def test_hitching_classification():
    assert GNT(N=2, Lh=[0, 0], L=[1, 1]).hitching == "SNT"
    assert GNT(N=2, Lh=[0.3, 0.2], L=[1, 1]).hitching == "nSNT"
    assert GNT(N=2, Lh=[0.3, 0.0], L=[1, 1]).hitching == "GNT"
