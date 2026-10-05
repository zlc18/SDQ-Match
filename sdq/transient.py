"""Transient Erlang-A cost used by the SDQ-Match matcher.

A boarding spot with c aircraft is an M/M/c+M queue whose state n is the
number of tourists in the system (in flight plus waiting). From an initial
state n0 the expected waiting time accumulated over a window of length H is

    V_H(c, a; n0) = E[ int_0^H (n(u) - c)^+ du | n(0) = n0 ],

and, by Little's law on the sample path, the expected number of abandonments
over the window is theta * V_H. A matching epoch has two phases. During the
first tau_rep steps only the aircraft already at the spot serve (c1); from
tau_rep on, aircraft repositioned now have arrived (c2 = c1 + incoming). The
two-phase cost of a spot is

    f(c1, c2) = V_tau(c1, a1; n0) + sum_n P_tau(c1, a1)[n0, n] * V_H(c2, a2; n),

computed exactly from the generator of the truncated birth-death chain with a
single matrix exponential per (c, a) cell.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class TransientCost:
    def __init__(self, mu: float, theta: float, tau: int, H2: int,
                 cmax: int, nmax: int | None = None, da: float = 0.05):
        self.mu, self.theta, self.tau, self.H2 = mu, theta, tau, H2
        self.cmax = cmax
        self.nmax = nmax if nmax is not None else cmax + 45
        self.da = da
        self._c1: dict = {}
        self._c2: dict = {}

    def _gen(self, c: int, lam: float) -> np.ndarray:
        M = self.nmax + 1
        n = np.arange(M)
        G = np.zeros((M, M))
        G[n[:-1], n[:-1] + 1] = lam
        death = np.minimum(n, c) * self.mu + np.maximum(n - c, 0) * self.theta
        G[n[1:], n[1:] - 1] = death[1:]
        G[n, n] = -G.sum(1)
        return G

    def _block(self, c: int, li: int, H: float):
        lam = li * self.da * self.mu
        G = self._gen(c, lam)
        M = G.shape[0]
        r = np.maximum(np.arange(M) - c, 0).astype(float)
        A = np.zeros((M + 1, M + 1))
        A[:M, :M] = G
        A[:M, M] = r
        E = expm(A * H)
        return E[:M, :M], E[:M, M]       # P_H, V_H

    def phase1(self, c: int, li: int):
        key = (c, li)
        if key not in self._c1:
            P, V = self._block(c, li, self.tau)
            self._c1[key] = (P.astype(np.float32), V.astype(np.float32))
        return self._c1[key]

    def phase2(self, c: int, li: int):
        key = (c, li)
        if key not in self._c2:
            _, V = self._block(c, li, self.H2)
            self._c2[key] = V.astype(np.float32)
        return self._c2[key]

    def li(self, a: float) -> int:
        return int(min(round(a / self.da), 10_000))

    def spot_cost(self, n0: int, c1: int, a1: float, c2: int, a2: float) -> float:
        n0 = min(n0, self.nmax)
        c1, c2 = min(c1, self.cmax), min(c2, self.cmax)
        P, V1 = self.phase1(c1, self.li(a1))
        V2 = self.phase2(c2, self.li(a2))
        return float(V1[n0] + P[n0] @ V2)

    def delta_costs(self, n0: int, committed: int, idle: int, a1: float, a2: float,
                    dmax: int) -> np.ndarray:
        """Cost f(d) for net change d = -idle..dmax of the spot's supply.

        d <= 0 removes idle aircraft now (c1 = c2 = committed + d); d > 0 adds
        aircraft that arrive after tau_rep (c1 = committed, c2 = committed + d).
        """
        n0 = min(n0, self.nmax)
        i1, i2 = self.li(a1), self.li(a2)
        out = np.empty(idle + dmax + 1)
        for k, d in enumerate(range(-idle, dmax + 1)):
            c1 = min(committed + min(d, 0), self.cmax)
            c2 = min(committed + d, self.cmax)
            P, V1 = self.phase1(c1, i1)
            V2 = self.phase2(c2, i2)
            out[k] = V1[n0] + P[n0] @ V2
        return out

    def block(self, c: int, li: int, L: int):
        """(P_L, V_L) for an arbitrary integer window length L (cached)."""
        key = (c, li, L)
        if key not in self._c1:
            P, V = self._block(c, li, L)
            self._c1[key] = (P.astype(np.float32), V.astype(np.float32))
        return self._c1[key]

    def delta_costs_inbound(self, n0: int, present: int, inbound: np.ndarray, idle: int,
                            a1: float, a2: float, dmax: int) -> np.ndarray:
        """Cost f(d) when some committed aircraft are still in flight to the spot.

        ``inbound`` holds the steps until each inbound aircraft lands. Phase 1 is
        split at those landing times, so an inbound aircraft serves only after it
        has arrived; ``present`` counts the idle and busy aircraft at the spot.
        """
        n0 = min(n0, self.nmax)
        i1, i2 = self.li(a1), self.li(a2)
        eta = np.sort(np.clip(np.asarray(inbound, dtype=int), 0, self.tau))
        cuts = sorted(set([0, self.tau] + [int(e) for e in eta]))
        segs = [(cuts[j], cuts[j + 1]) for j in range(len(cuts) - 1)]
        committed = present + len(eta)
        out = np.empty(idle + dmax + 1)
        for k, d in enumerate(range(-idle, dmax + 1)):
            dm = min(d, 0)
            p = None
            cost = 0.0
            for (s0, s1) in segs:
                c = min(present + dm + int((eta <= s0).sum()), self.cmax)
                P, V = self.block(c, i1, s1 - s0)
                if p is None:
                    cost += V[n0]
                    p = P[n0].astype(np.float64)
                else:
                    cost += p @ V
                    p = p @ P
            c2 = min(committed + d, self.cmax)
            out[k] = cost + p @ self.phase2(c2, i2)
        return out


def convex_minorant_increments(f: np.ndarray) -> np.ndarray:
    """Marginal costs of the lower convex envelope of f (non-decreasing)."""
    inc = np.diff(f)
    # pool-adjacent-violators: make increments non-decreasing, preserving sums
    vals, wts = [], []
    for x in inc:
        vals.append(x)
        wts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            w = wts[-2] + wts[-1]
            vals.pop(); wts.pop()
            vals[-1], wts[-1] = v, w
    out = np.concatenate([[v] * w for v, w in zip(vals, wts)]) if vals else inc
    return out
