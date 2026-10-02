"""The evaluation protocol: laps, lead-in, run-out, per-segment spans and divergence."""

import numpy as np
import pytest

from gntpf import GNT, make_path
from gntpf.sim import Watchdog, evaluation_masks


def test_closed_path_is_repeated_and_carries_its_evaluation_span():
    one = make_path("rounded_rect")
    two = make_path("rounded_rect", laps=2, lead=5.0, runout=3.0)
    assert two.eval_start == pytest.approx(5.0)
    assert two.eval_end - two.eval_start == pytest.approx(2 * one.length, rel=1e-3)
    assert two.length == pytest.approx(5.0 + 2 * one.length + 3.0, abs=0.05)
    # the joins add no curvature beyond that of the path itself
    assert np.max(np.abs(two.curvature())) == pytest.approx(np.max(np.abs(one.curvature())), rel=1e-3)


def test_open_path_cannot_be_lapped():
    with pytest.raises(ValueError):
        make_path("agricultural", laps=2)


def test_masks_follow_each_segment_into_and_out_of_the_span():
    p = make_path("rounded_rect", laps=2, lead=5.0, runout=3.0)
    s_proj = np.array([[1.0, 6.0, p.eval_end + 1.0],     # tractor: before, inside, after
                       [0.5, 4.0, p.eval_end - 1.0]])    # trailer: before, before, inside
    m, _ = evaluation_masks(p, s_proj)
    assert m.tolist() == [[False, True, False], [False, False, True]]


def test_watchdog_fires_only_after_a_sustained_lag():
    model = GNT(N=2, Lh=[0.342, 0.0], L=[1.08, 0.78], Ts=0.05)
    w = Watchdog(model, Ts=0.05)
    lag = 2.5 * model.total_length
    hold = int(round(5.0 / 0.05))                                         # 5 s of lag
    assert not any(w.update(k, 0.05, 100.0, 100.0 - lag) for k in range(hold - 1))
    assert w.update(hold - 1, 0.05, 100.0, 100.0 - lag)
    assert w.t_diverged == pytest.approx(0.0, abs=1e-9)                  # the lag began at k = 0
    w2 = Watchdog(model, Ts=0.05)
    assert not any(w2.update(k, 0.05, 100.0, 100.0 - 0.5 * model.total_length) for k in range(400))
