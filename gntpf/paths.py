"""Nominal paths and the moving local polynomial approximation.

The controller never sees the global path.  At every sampling instant a window of
the nominal path around the vehicle's virtual progress point is refitted by a
polynomial of low degree, and the reference generator works entirely in the
window's arc-length coordinate ``s``.  This keeps the reference-generation NLP
small and its decision variables ``lambda_i`` interpretable: they are arc lengths
along the path, ordered from the tractor backwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import atan2c

__all__ = ["Path", "LocalFit", "make_path", "PATHS"]


# --------------------------------------------------------------------------- #
#  Dense paths
# --------------------------------------------------------------------------- #
def _circle(n=4000, R=4.0, cx=5.5, cy=4.5):
    g = np.linspace(0, 2 * np.pi, n)
    return np.vstack([R * np.cos(g) + cx, R * np.sin(g) + cy])


def _lemniscate(n=6000, cx=5.5, cy=4.5):
    g = np.linspace(0, 2 * np.pi, n)
    d = np.sin(g) ** 2 + 1.0
    return np.vstack([np.sqrt(32) * np.cos(g) / d + cx, np.sqrt(200) * np.sin(g) * np.cos(g) / d + cy])


def _square(n=8000, a=3.0, b=1.0, cx=7.0, cy=4.5):
    """Superellipse-like squared path: right-angle corners are kinematically
    incompatible with the chain, which is the point of the test."""
    g = np.linspace(0, 2 * np.pi, n)
    cs, sn = np.cos(g), np.sin(g)
    x = a * np.sqrt(cs**2) * cs + b * np.sqrt(sn**2) * sn + cx
    y = a * np.sqrt(cs**2) * cs - b * np.sqrt(sn**2) * sn + cy
    return np.vstack([x, y])


def _rounded_rect(n=6000, w=16.0, h=10.0, r=1.2, cx=0.0, cy=0.0):
    """Rectangle with rounded corners -- a headland-turn geometry."""
    w2, h2 = w / 2 - r, h / 2 - r
    segs = []
    straights = [((-w2, -h / 2), (w2, -h / 2)), ((w / 2, -h2), (w / 2, h2)),
                 ((w2, h / 2), (-w2, h / 2)), ((-w / 2, h2), (-w / 2, -h2))]
    corners = [((w2, -h2), -np.pi / 2, 0.0), ((w2, h2), 0.0, np.pi / 2),
               ((-w2, h2), np.pi / 2, np.pi), ((-w2, -h2), np.pi, 1.5 * np.pi)]
    for i in range(4):
        (p0, p1) = straights[i]
        t = np.linspace(0, 1, n // 8)
        segs.append(np.vstack([p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])]))
        (c, a0, a1) = corners[i]
        a = np.linspace(a0, a1, n // 8)
        segs.append(np.vstack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)]))
    xy = np.hstack(segs)
    return xy + np.array([[cx], [cy]])


def _omega_turn(p1, h1, p2, h2, r, ds=0.002):
    """Three-arc omega ("bulb") headland turn between two antiparallel rows.

    A single arc of radius ``r`` cannot join two rows whose spacing is less than
    ``2r``.  The omega turn swings away from the next row, loops round in the
    headland, and comes back into the next row: three tangent arcs of radius
    ``r``, the outer two turning one way and the middle one the other, with a net
    change of heading of 180 degrees.  The whole turn lies in the headland,
    beyond the ends of the rows, and is tangent-continuous with both rows.
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    h1, h2 = np.asarray(h1, float), np.asarray(h2, float)
    left = lambda h: np.array([-h[1], h[0]])            # left normal of a heading
    e = p2 - p1
    s = -1.0 if float(e @ left(h1)) > 0 else 1.0        # first turn away from the next row
    cA = p1 + r * s * left(h1)                           # outer arcs: sense s
    cC = p2 + r * s * left(h2)
    mid, dist = 0.5 * (cA + cC), float(np.linalg.norm(cC - cA))
    if dist >= 4 * r:
        raise ValueError("omega turn needs the row spacing below 2r")
    half = np.sqrt((2 * r) ** 2 - (dist / 2) ** 2)
    cB = mid + half * h1                                 # middle arc: in the headland

    def ang(v):
        return float(np.arctan2(v[1], v[0]))

    def arc(c, a0, a1, sense):
        d = (a1 - a0) % (2 * np.pi)
        sweep = d if sense > 0 else d - 2 * np.pi
        n = max(int(abs(sweep) * r / ds), 2)
        a = np.linspace(a0, a0 + sweep, n)
        return np.vstack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])

    tAB = 0.5 * (cA + cB)                                # tangent points
    tBC = 0.5 * (cB + cC)
    segA = arc(cA, ang(p1 - cA), ang(tAB - cA), s)
    segB = arc(cB, ang(tAB - cB), ang(tBC - cB), -s)
    segC = arc(cC, ang(tBC - cC), ang(p2 - cC), s)
    return np.hstack([segA, segB[:, 1:], segC[:, 1:]])


def _agricultural(r=2.0, l=7.0, d=1.5, delta=0.002, rows=5):
    """Boustrophedon field path with omega-shaped headland turns.

    ``rows`` parallel crop rows of length ``l`` spaced ``d`` apart, joined by
    omega turns of radius ``r``.  Because the row spacing is smaller than the
    turn diameter (``d < 2r``) the machine cannot turn straight from one row into
    the next; it must swing out and come back, which is the "keyhole" turn that
    makes this geometry the demanding one in practice.

    The turns are tangent-continuous with the rows, so the path is G1 and its
    curvature is bounded by ``1/r``.  The rows run horizontally, as in the
    MATLAB version.
    """
    if d >= 2 * r:
        raise ValueError("agricultural turn assumes d < 2r (an omega turn)")
    segs = []
    y = np.arange(0.0, l + delta, delta)
    segs.append(np.vstack([np.zeros_like(y), y]))
    for k in range(rows - 1):
        up = (k % 2 == 0)
        h1 = np.array([0.0, 1.0]) if up else np.array([0.0, -1.0])
        p1 = segs[-1][:, -1]
        p2 = np.array([p1[0] + d, p1[1]])
        turn = _omega_turn(p1, h1, p2, -h1, r, ds=delta)
        segs.append(turn[:, 1:])
        p = segs[-1][:, -1]
        y = (np.arange(p[1], -delta, -delta) if up
             else np.arange(p[1], l + delta, delta))
        segs.append(np.vstack([np.full_like(y, p[0]), y])[:, 1:])
    xy = np.hstack(segs)
    th = -np.pi / 2
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return R @ xy


def _corridor(n=6000, length=26.0, amp=2.6, period=13.0):
    """A sinuous lane, used together with :func:`corridor_bounds` for the
    narrow-road study."""
    s = np.linspace(0.0, length, n)
    return np.vstack([s, amp * np.sin(2 * np.pi * s / period)])


def corridor_bounds(xy, half_width):
    """Offset a centre-line by ``+/- half_width`` along its normal."""
    d = np.gradient(xy, axis=1)
    nrm = np.linalg.norm(d, axis=0, keepdims=True)
    nrm[nrm < 1e-12] = 1e-12
    t = d / nrm
    nvec = np.vstack([-t[1], t[0]])
    return xy + half_width * nvec, xy - half_width * nvec


PATHS = {
    "agricultural": _agricultural,
    "circle": _circle,
    "lemniscate": _lemniscate,
    "square": _square,
    "rounded_rect": _rounded_rect,
    "corridor": _corridor,
}


# Paths that close on themselves and can therefore be traversed several times.
CLOSED_PATHS = {"circle", "lemniscate", "square", "rounded_rect"}


def _arc_length(xy):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=1)))])


def _straight(p, direction, length, ds=0.01):
    """Points from ``p`` along the unit ``direction`` for ``length`` metres, excluding ``p``."""
    n = max(int(np.ceil(length / ds)), 1)
    t = np.linspace(0.0, length, n + 1)[1:]
    return p[:, None] + direction[:, None] * t[None, :]


def make_path(name: str, laps: int = 1, lead: float = 0.0, runout: float = 0.0,
              **kw) -> "Path":
    """Build a named path.

    ``laps`` repeats a closed path; ``lead`` prepends a straight approach along the
    initial tangent; ``runout`` extends the path after its end -- the beginning of
    the next lap for a closed path, a straight along the final tangent otherwise.
    The evaluation span of the returned path is the nominal part, ``laps`` laps
    long, between the lead-in and the run-out.
    """
    if name not in PATHS:
        raise KeyError(f"unknown path {name!r}; available: {sorted(PATHS)}")
    xy = np.asarray(PATHS[name](**kw), dtype=float)
    closed = name in CLOSED_PATHS
    if laps > 1 and not closed:
        raise ValueError(f"path {name!r} is open and cannot be traversed {laps} times")
    if closed and np.hypot(*(xy[:, -1] - xy[:, 0])) < 1e-9:
        loop = xy[:, :-1]                    # drop the closing duplicate
    else:
        loop = xy
    body = np.hstack([loop] * laps + ([xy[:, :1]] if closed else []))
    nominal = float(_arc_length(body)[-1])

    parts = []
    if lead > 0:
        d0 = body[:, 1] - body[:, 0]
        d0 = d0 / np.linalg.norm(d0)
        parts.append(_straight(body[:, 0], -d0, lead)[:, ::-1])
    parts.append(body)
    if runout > 0:
        if closed:
            s_loop = _arc_length(loop)
            k = int(np.searchsorted(s_loop, runout)) + 1
            parts.append(loop[:, 1:k + 1])
        else:
            d1 = body[:, -1] - body[:, -2]
            d1 = d1 / np.linalg.norm(d1)
            parts.append(_straight(body[:, -1], d1, runout))
    full = np.hstack(parts)
    return Path(full, name=name, eval_start=lead, eval_end=lead + nominal)


# --------------------------------------------------------------------------- #
#  Local polynomial fit
# --------------------------------------------------------------------------- #
@dataclass
class LocalFit:
    """Polynomial approximation of a path window, in local arc length."""

    cx: np.ndarray       # descending-power coefficients for x(s)
    cy: np.ndarray
    cdx: np.ndarray      # derivative coefficients dx/ds
    cdy: np.ndarray
    s_len: float         # local arc length spanned by the window
    s0_global: float     # global arc length at s = 0

    def xy(self, s):
        return np.array([np.polyval(self.cx, s), np.polyval(self.cy, s)])

    def tangent(self, s):
        return np.array([np.polyval(self.cdx, s), np.polyval(self.cdy, s)])

    def heading(self, s, previous=0.0):
        d = self.tangent(s)
        return atan2c(d[1], d[0], previous)

    def rms_error(self, xy_dense, s_dense) -> float:
        """Fit residual, for reporting how well the window is represented."""
        approx = np.vstack([np.polyval(self.cx, s_dense), np.polyval(self.cy, s_dense)])
        return float(np.sqrt(np.mean(np.sum((approx - xy_dense) ** 2, axis=0))))


class Path:
    """A dense nominal path with arc-length bookkeeping and windowed refitting."""

    def __init__(self, xy: np.ndarray, name: str = "path",
                 eval_start: float = 0.0, eval_end: float | None = None):
        self.xy = np.asarray(xy, dtype=float)
        if self.xy.shape[0] != 2:
            raise ValueError("path coordinates must be a 2 x M array")
        # Drop repeated samples.  A duplicated point spans no distance, so the
        # heading across it is undefined and the curvature computed from it is
        # arbitrarily large; left in, a handful of them dominate every
        # curvature-resolved metric taken on the path.
        if self.xy.shape[1] > 1:
            keep = np.concatenate([[True], np.hypot(*np.diff(self.xy, axis=1)) > 1e-9])
            self.xy = self.xy[:, keep]
        self.name = name
        d = np.diff(self.xy, axis=1)
        seg = np.hypot(d[0], d[1])
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.s[-1])
        # Arc-length span over which a run is evaluated.  A lead-in before it
        # absorbs the start-up transient, and a run-out after it lets the last
        # trailer finish the span before the progress point reaches the end.
        self.eval_start = float(eval_start)
        self.eval_end = self.length if eval_end is None else float(min(eval_end, self.length))

    # ------------------------------------------------------------------ utils
    def at(self, s):
        """Point at global arc length ``s`` (clamped to the path)."""
        s = np.clip(s, 0.0, self.length)
        return np.array([np.interp(s, self.s, self.xy[0]), np.interp(s, self.s, self.xy[1])])

    def project(self, p, s_guess=None, window=None) -> float:
        """Arc length of the closest path point to ``p``.

        When ``s_guess`` and ``window`` are given the search is restricted to
        ``s_guess +/- window``, which keeps the projection unique on paths that
        pass close to themselves.
        """
        p = np.asarray(p).reshape(2)
        if s_guess is None or window is None:
            idx = slice(None)
            base = 0
        else:
            lo = np.searchsorted(self.s, s_guess - window)
            hi = np.searchsorted(self.s, s_guess + window)
            lo, hi = max(0, lo), min(self.s.size, max(hi, lo + 2))
            idx = slice(lo, hi)
            base = lo
        d2 = np.sum((self.xy[:, idx] - p[:, None]) ** 2, axis=0)
        return float(self.s[base + int(np.argmin(d2))])

    def deviation(self, p, s_guess=None, window=None) -> float:
        """Euclidean distance from ``p`` to the path (the off-tracking measure)."""
        s = self.project(p, s_guess, window)
        return float(np.linalg.norm(self.at(s) - np.asarray(p).reshape(2)))

    def tangent(self, s) -> np.ndarray:
        """Unit tangent of the dense path at arc length ``s``."""
        s = float(np.clip(s, 0.0, self.length))
        h = max(self.length * 1e-4, 1e-3)
        a, b = self.at(max(s - h, 0.0)), self.at(min(s + h, self.length))
        d = b - a
        n = np.linalg.norm(d)
        return d / n if n > 1e-12 else np.array([1.0, 0.0])

    def signed_deviation(self, p, s_guess=None, window=None) -> float:
        """Lateral offset of ``p`` from the path, positive to the left.

        The sign is what distinguishes a vehicle that swings wide on one side
        from one that straddles the path.  The width of the corridor a vehicle
        needs is the span between its left-most and right-most excursions, which
        cannot be recovered from unsigned distances.
        """
        s = self.project(p, s_guess, window)
        t = self.tangent(s)
        d = np.asarray(p).reshape(2) - self.at(s)
        return float(-t[1] * d[0] + t[0] * d[1])

    # ------------------------------------------------------------- curvature
    def curvature(self) -> np.ndarray:
        """Signed-magnitude curvature of the dense path, one value per sample."""
        d1 = np.gradient(self.xy, axis=1)
        d2 = np.gradient(d1, axis=1)
        num = np.abs(d1[0] * d2[1] - d1[1] * d2[0])
        den = (d1[0] ** 2 + d1[1] ** 2) ** 1.5 + 1e-15
        return num / den

    def curved_mask(self, s_values, k_threshold: float = 0.15) -> np.ndarray:
        """Boolean mask selecting the arc lengths that lie in curved sections.

        Off-tracking is a non-issue on a straight; reporting an average over a
        path that is mostly straight hides the behaviour that actually matters.
        """
        k = self.curvature()
        ks = np.interp(np.asarray(s_values), self.s, k)
        return ks > k_threshold

    # -------------------------------------------------------------- windowing
    def local_fit(self, s_centre: float, back: float, fwd: float, order: int = 7) -> LocalFit:
        """Fit polynomials to the window ``[s_centre - back, s_centre + fwd]``.

        The window is parameterised by its own arc length, so ``s = back`` is the
        centre point.  Returns a :class:`LocalFit`.
        """
        s_lo = max(0.0, s_centre - back)
        s_hi = min(self.length, s_centre + fwd)
        if s_hi - s_lo < 1e-6:
            raise ValueError("degenerate path window")
        lo = int(np.searchsorted(self.s, s_lo))
        hi = int(np.searchsorted(self.s, s_hi))
        hi = max(hi, lo + order + 2)
        hi = min(hi, self.s.size)
        lo = min(lo, hi - order - 2)
        lo = max(lo, 0)

        xy = self.xy[:, lo:hi]
        sl = self.s[lo:hi] - self.s[lo]
        cx = np.polyfit(sl, xy[0], order)
        cy = np.polyfit(sl, xy[1], order)
        return LocalFit(cx, cy, np.polyder(cx), np.polyder(cy), float(sl[-1]), float(self.s[lo]))
