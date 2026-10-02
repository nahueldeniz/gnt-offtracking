"""Process-level parallelism for the simulation studies.

Every run in a sweep is independent, so the sweeps are embarrassingly parallel.
The number of worker processes is taken from the environment variable
``GNT_JOBS`` (default 1, i.e. serial), so that a study can be run serially when
solve times are being measured and in parallel when they are not.

Workers must be given plain data.  A ``GNT`` model holds CasADi objects and is
not picklable, so each job builds its own model from the arguments it is sent.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

__all__ = ["n_jobs", "pmap"]


def n_jobs() -> int:
    """Worker processes requested through ``GNT_JOBS`` (1 = serial)."""
    try:
        return max(1, int(os.environ.get("GNT_JOBS", "1")))
    except ValueError:
        return 1


def pmap(func, args):
    """``[func(*a) for a in args]``, evaluated in ``n_jobs()`` processes.

    Order is preserved.  With ``GNT_JOBS=1`` the jobs run in this process, which
    is what the timing studies need: measured solve times are only meaningful
    when the machine is not oversubscribed.
    """
    args = list(args)
    jobs = min(n_jobs(), len(args))
    if jobs <= 1:
        return [func(*a) for a in args]
    # Each worker is restricted to one BLAS thread; the parallelism is here.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(_call, [(func, a) for a in args]))


def _call(job):
    func, args = job
    return func(*args)
