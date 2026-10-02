"""The reachability residual e_r of Eq. (11) is recorded, and is small where the
path is attainable."""

import numpy as np

from gntpf import GNT, SimConfig, make_path, simulate


def test_reachability_residual_is_recorded_and_small_on_a_circle():
    model = GNT(N=2, Lh=[0.342, 0.0], L=[1.08, 0.78], Ts=0.05)
    path = make_path("circle", R=8.0, cx=9.0, cy=9.0)   # constant, attainable curvature
    res = simulate(model, path, SimConfig(sigma=1.0, t_final=6.0, seed=0))

    assert res.reach is not None
    assert res.reach.shape == (model.nq, res.t.size)
    s = res.reach_summary(skip=0.3)
    # a reference reachable from the previous one to within a few millimetres,
    # against a progress of sigma * Ts = 50 mm per step
    assert s["pos_m"]["all"]["max"] < 5e-3
    assert s["ang_deg"]["all"]["max"] < 0.5
