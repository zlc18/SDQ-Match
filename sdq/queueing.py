"""Exact multi-server waiting-time laws used by the matcher and the baselines.

Erlang-A (M/M/c+M): Poisson arrivals at rate lambda, c exponential servers at
rate mu, exponential patience at rate theta. The stationary distribution of
the birth-death chain gives the mean number waiting L_q(c, a), a = lambda/mu.
By Little's law L_q is the long-run waiting time accrued per unit time and
theta * L_q is the abandonment rate, so both cost terms are proportional to
L_q. For theta -> 0 and a < c it reduces to the Erlang-C law
L_q = C(c, a) * a / (c - a).
"""
from __future__ import annotations

import math

import numpy as np


def erlang_a_lq(c: int, a: float, r: float, kmax: int | None = None) -> float:
    """Mean queue length of M/M/c+M with offered load a and r = theta/mu."""
    if a <= 0:
        return 0.0
    if kmax is None:
        kmax = c + int(10 + 8 * a / max(r, 1e-3) ** 0.5 + 6 * a / max(r, 1e-3))
        kmax = min(kmax, c + 4000)
    # log of unnormalised probabilities
    logp = np.empty(kmax + 1)
    logp[0] = 0.0
    la = math.log(a)
    for n in range(1, kmax + 1):
        death = min(n, c) + max(n - c, 0) * r
        logp[n] = logp[n - 1] + la - math.log(death)
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    n = np.arange(kmax + 1)
    return float((np.maximum(n - c, 0) * p).sum())


def erlang_c_lq(c: int, a: float) -> float:
    """Erlang-C mean queue length (infinite when a >= c)."""
    if a <= 0:
        return 0.0
    if c <= a:
        return math.inf
    s = 0.0
    term = 1.0
    for k in range(c):
        if k > 0:
            term *= a / k
        s += term
    top = term * a / c * c / (c - a)  # a^c/c! * c/(c-a)
    pw = top / (s + top)
    return pw * a / (c - a)


class ErlangATable:
    """Pre-tabulated L_q(c, a) on a load grid, linear interpolation in a."""

    def __init__(self, cmax: int, r: float, amax: float, da: float = 0.02):
        self.cmax, self.r, self.da = cmax, r, da
        self.grid = np.arange(0.0, amax + da, da)
        tab = np.zeros((cmax + 2, len(self.grid)))
        for c in range(cmax + 2):
            for i, a in enumerate(self.grid):
                tab[c, i] = erlang_a_lq(c, a, r)
        self.tab = tab
        self.amax = amax

    def lq(self, a: np.ndarray) -> np.ndarray:
        """Return L_q for all c in 0..cmax+1 at loads a (shape S) -> (S, cmax+2)."""
        a = np.asarray(a, dtype=float)
        x = np.clip(a / self.da, 0, len(self.grid) - 1.000001)
        i = x.astype(int)
        f = x - i
        lo = self.tab[:, i]
        hi = self.tab[:, np.minimum(i + 1, len(self.grid) - 1)]
        out = (lo * (1 - f) + hi * f).T
        # loads beyond the table: L_q grows linearly in a for c fixed (pure abandonment regime)
        over = a > self.amax
        if over.any():
            out[over] += (a[over, None] - self.amax) / self.r
        return out


_TABLES: dict = {}


def get_table(cmax: int, r: float, amax: float) -> ErlangATable:
    key = (cmax, round(r, 6), amax)
    if key not in _TABLES:
        _TABLES[key] = ErlangATable(cmax, r, amax)
    return _TABLES[key]


def greedy_allocate(gains: np.ndarray, N: int, floor: np.ndarray | None = None) -> np.ndarray:
    """Marginal-gain water-filling.

    gains[s, c] is the cost reduction of giving spot s its (c+1)-th server and
    must be non-increasing in c (discrete convexity of the cost). Returns the
    integer allocation that maximises the total gain with sum N, which is the
    exact minimiser of the separable convex cost.
    """
    S = gains.shape[0]
    c = np.zeros(S, dtype=int) if floor is None else floor.astype(int).copy()
    rem = N - c.sum()
    cmax = gains.shape[1]
    while rem > 0:
        idx = np.minimum(c, cmax - 1)
        g = gains[np.arange(S), idx]
        g = np.where(c >= cmax, -np.inf, g)
        s = int(np.argmax(g))
        c[s] += 1
        rem -= 1
    return c
