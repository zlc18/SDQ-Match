"""Matching policies: baselines and the queueing core of SDQ-Match.

Every policy implements ``reset(env)`` and ``act(obs) -> target allocation``
(a non-negative integer vector summing to N). All policies act through the
same environment interface, so they differ only in the matching decision.
"""
from __future__ import annotations

import numpy as np

from .forecast import Forecaster, historical_profile
from .queueing import get_table, greedy_allocate


def largest_remainder(weights: np.ndarray, N: int) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=float), 0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    x = N * w / w.sum()
    c = np.floor(x).astype(int)
    rem = N - c.sum()
    if rem > 0:
        order = np.argsort(-(x - c), kind="stable")
        c[order[:rem]] += 1
    return c


class Policy:
    name = "policy"
    uses_forecast = False

    def __init__(self, cfg, forecaster: Forecaster | None = None):
        self.cfg = cfg
        self.fc = forecaster

    def reset(self, env, seed: int = 0):
        self.env = env
        if self.fc is not None:
            self.fc.reset(env, seed)

    def act(self, obs):
        raise NotImplementedError


# --------------------------------------------------------------------- simple
class RandomPolicy(Policy):
    name = "Random"

    def reset(self, env, seed: int = 0):
        super().reset(env, seed)
        self.rng = np.random.default_rng(777 + seed)

    def act(self, obs):
        return largest_remainder(self.rng.dirichlet(np.ones(self.cfg.S)), self.cfg.N)


class GreedyPolicy(Policy):
    """Supply proportional to the current queue plus the work arriving now."""
    name = "Greedy"

    def act(self, obs):
        w = obs["q"] + obs["recent_rate"] * self.cfg.mean_service + 1e-6
        return largest_remainder(w, self.cfg.N)


class JSQPolicy(Policy):
    """Join-shortest-queue for supply: the next aircraft goes to the longest queue per server."""
    name = "JSQ"

    def act(self, obs):
        S, N = self.cfg.S, self.cfg.N
        q = obs["q"].astype(float) + 1e-3 * obs["committed"]   # tie-break: stay put
        c = np.arange(N + 1)
        gains = q[:, None] / (c[None, :] + 1.0)
        return greedy_allocate(gains, N)


class StaticErlangC(Policy):
    """Static square-root staffing on each spot's average daily offered load."""
    name = "Erlang-C"

    def __init__(self, cfg, forecaster=None, beta: float = 0.5):
        super().__init__(cfg, forecaster)
        self.beta = beta
        hist = historical_profile(cfg)
        a = hist.mean(0) * cfg.mean_service
        self.target = largest_remainder(a + beta * np.sqrt(a), cfg.N)

    def act(self, obs):
        return self.target.copy()


# --------------------------------------------------------- Erlang-A families
class _ErlangPolicy(Policy):
    def __init__(self, cfg, forecaster=None):
        super().__init__(cfg, forecaster)
        r = cfg.mean_service / cfg.mean_patience
        self.table = get_table(cfg.N, r, amax=max(12.0, 1.5 * cfg.N))

    def erlang_gains(self, a: np.ndarray) -> np.ndarray:
        lq = self.table.lq(a)                     # (S, N+2)
        g = lq[:, :-1] - lq[:, 1:]                # gain of the (c+1)-th server
        # enforce monotonicity against interpolation noise (convexity holds exactly)
        return np.minimum.accumulate(np.maximum(g, 0.0), axis=1)


class WhittlePolicy(_ErlangPolicy):
    """Index policy: each extra server goes to the spot with the largest marginal
    holding-cost reduction (Erlang-A queue length) at the current backlog.

    rate="observed": the arrival rate is the one observed over the last 15 steps
    (no forecast; the non-anticipatory index baseline).
    rate="forecast": the rate is the forecast for the coming epoch (Whittle-F),
    a stronger variant that shares SDQ-Match's forecaster.
    """
    name = "Whittle"
    uses_forecast = True

    def __init__(self, cfg, forecaster=None, h: float = 40.0, rate: str = "observed"):
        super().__init__(cfg, forecaster)
        self.h, self.rate = h, rate
        if rate == "forecast":
            self.name = "Whittle-F"

    def act(self, obs):
        t = obs["t"]
        if self.rate == "forecast":
            lam = self.fc.mean_rate(t, t + self.cfg.kappa)
        else:
            lam = obs["recent_rate"]
        a = (lam + obs["q"] / self.h) * self.cfg.mean_service
        return greedy_allocate(self.erlang_gains(a), self.cfg.N)


class LagSIPP(_ErlangPolicy):
    """Lagged stationary-independent-period-by-period staffing (Green, Kolesar & Soares, 2001).

    The offered load of the service window that a moved aircraft can reach,
    [t+tau_rep, t+tau_rep+kappa), is the infinite-server load
    m(u) = sum_k lambda(u-k) P(S > k), and the fleet is split by the exact
    Erlang-A allocation of that load.
    """
    name = "Lag-SIPP"
    uses_forecast = True

    def act(self, obs):
        cfg = self.cfg
        t = obs["t"]
        u = np.arange(t + cfg.tau_rep, t + cfg.tau_rep + cfg.kappa)
        k = np.arange(0, int(6 * cfg.mean_service))
        surv = (1 - cfg.mu) ** k                     # P(service > k), geometric
        ts = (u[:, None] - k[None, :]).ravel()
        lam = self.fc.rate_at(ts).reshape(len(u), len(k), cfg.S)
        m = (lam * surv[None, :, None]).sum(1).mean(0)    # offered load in servers
        return greedy_allocate(self.erlang_gains(m), cfg.N)


# ---------------------------------------------------------- metaheuristics
class _FluidMPC(Policy):
    """Shared objective of the PSO and GA matchers: a fluid look-ahead of the
    queues over ``horizon`` steps under the current arrival rate, including the
    reposition delay of moved aircraft and patience-driven abandonment."""
    uses_forecast = True

    def __init__(self, cfg, forecaster, horizon: int = 20, rate: str = "observed"):
        super().__init__(cfg, forecaster)
        self.horizon, self.rate = horizon, rate

    def reset(self, env, seed: int = 0):
        super().reset(env, seed)
        self.rng = np.random.default_rng(4242 + seed)

    def cost(self, C: np.ndarray, obs, lam: np.ndarray) -> np.ndarray:
        """C: (P, S) candidate allocations -> (P,) fluid cost (waiting + abandonment)."""
        cfg = self.cfg
        idle, committed = obs["idle"], obs["committed"]
        # aircraft that must move arrive after tau_rep; others are in place now
        give = np.maximum(committed[None, :] - C, 0)
        take = np.maximum(C - committed[None, :], 0)
        movable = np.minimum(give, idle[None, :]).sum(1)
        feasible_in = np.minimum(take.sum(1), movable)
        frac_in = np.where(take.sum(1) > 0, feasible_in / np.maximum(1, take.sum(1)), 0)
        now = np.minimum(committed[None, :], C).astype(float)
        later = take * frac_in[:, None]
        q = np.repeat(obs["q"][None, :].astype(float), len(C), 0)
        cost = np.zeros(len(C))
        for k in range(self.horizon):
            cap = now + (later if k >= cfg.tau_rep else 0.0)
            ab = q * cfg.theta
            q = np.maximum(q + lam[None, :] - cap * cfg.mu - ab, 0.0)
            cost += q.sum(1) * cfg.beta2 + ab.sum(1) * cfg.beta3
        return cost

    def rates(self, obs):
        """observed: rate seen over the last 15 steps (the non-anticipatory
        baseline); forecast: the forecaster's rate over the look-ahead window."""
        if self.rate == "forecast":
            t = obs["t"]
            return self.fc.mean_rate(t, t + self.horizon)
        return obs["recent_rate"]


def _repair(x: np.ndarray, N: int) -> np.ndarray:
    return np.array([largest_remainder(row, N) for row in x])


class PSOPolicy(_FluidMPC):
    name = "PSO"

    def __init__(self, cfg, forecaster, horizon=20, particles=20, iters=20, rate="observed"):
        super().__init__(cfg, forecaster, horizon, rate)
        if rate == "forecast":
            self.name = "PSO-F"
        self.P, self.I = particles, iters

    def act(self, obs):
        S, N = self.cfg.S, self.cfg.N
        lam = self.rates(obs)
        rng = self.rng
        x = rng.dirichlet(np.ones(S), self.P) * N
        x[0] = obs["committed"]
        v = np.zeros_like(x)
        f = self.cost(_repair(x, N), obs, lam)
        pb, pbf = x.copy(), f.copy()
        g = pb[np.argmin(pbf)].copy()
        for _ in range(self.I):
            r1, r2 = rng.random(x.shape), rng.random(x.shape)
            v = 0.7 * v + 1.5 * r1 * (pb - x) + 1.5 * r2 * (g - x)
            x = np.clip(x + v, 0, N)
            f = self.cost(_repair(x, N), obs, lam)
            better = f < pbf
            pb[better], pbf[better] = x[better], f[better]
            g = pb[np.argmin(pbf)].copy()
        return largest_remainder(g, N)


class GAPolicy(_FluidMPC):
    name = "GA"

    def __init__(self, cfg, forecaster, horizon=20, pop=20, gens=20, rate="observed"):
        super().__init__(cfg, forecaster, horizon, rate)
        if rate == "forecast":
            self.name = "GA-F"
        self.pop, self.gens = pop, gens

    def act(self, obs):
        S, N = self.cfg.S, self.cfg.N
        lam = self.rates(obs)
        rng = self.rng
        P = _repair(rng.dirichlet(np.ones(S), self.pop) * N, N)
        P[0] = obs["committed"]
        f = self.cost(P, obs, lam)
        for _ in range(self.gens):
            elite = P[np.argsort(f)[: max(2, self.pop // 5)]]
            # tournament selection
            i, j = rng.integers(0, self.pop, (2, self.pop))
            parents = np.where((f[i] < f[j])[:, None], P[i], P[j])
            mates = parents[rng.permutation(self.pop)]
            mask = rng.random((self.pop, S)) < 0.5
            child = np.where(mask, parents, mates).astype(float)
            child = _repair(child, N)
            # unit-move mutation
            mut = rng.random(self.pop) < 0.3
            for k in np.flatnonzero(mut):
                src = rng.choice(np.flatnonzero(child[k] > 0))
                dst = rng.integers(S)
                child[k, src] -= 1
                child[k, dst] += 1
            child[: len(elite)] = elite
            P = child
            f = self.cost(P, obs, lam)
        return P[np.argmin(f)]


# ------------------------------------------------------- SDQ-Match (core)
class SDQCore(_ErlangPolicy):
    """Queueing core of SDQ-Match.

    Effective offered load of spot s at the lead time,
        a_s = gamma * lambda_hat_s[t+tau, t+tau+H) / mu + q_s / (mu * h),
    optionally rescaled per spot by a learned correction exp(delta_s); the fleet
    is split by exact marginal-wait water-filling on the Erlang-A queue length,
    with a stay bonus eta for aircraft already committed to a spot (reposition cost).
    """
    name = "SDQ-core"
    uses_forecast = True

    def __init__(self, cfg, forecaster, gamma=1.0, h=20.0, eta=0.0, H=None,
                 lead=True, exact=True, coverage=True):
        super().__init__(cfg, forecaster)
        self.gamma, self.h, self.eta = gamma, h, eta
        self.H = H if H is not None else cfg.kappa
        self.lead, self.exact, self.coverage = lead, exact, coverage

    def loads(self, obs):
        cfg = self.cfg
        t = obs["t"]
        t0 = t + cfg.tau_rep if self.lead else t
        lam = self.fc.mean_rate(t0, t0 + self.H)
        return (self.gamma * lam + obs["q"] / self.h) * cfg.mean_service

    def allocate(self, a, obs):
        cfg = self.cfg
        if self.exact:
            g = self.erlang_gains(a)
        else:
            # pressure-per-server surrogate pi/(c+1) (ablation), gain pi/((c+1)(c+2))
            c = np.arange(cfg.N + 1)
            g = a[:, None] / ((c[None, :] + 1.0) * (c[None, :] + 2.0))
        if self.eta > 0:
            c = np.arange(g.shape[1])
            stay = (c[None, :] < obs["committed"][:, None]).astype(float)
            g = g + self.eta * stay
        floor = None
        if self.coverage:
            floor = ((obs["q"] > 0) | (a > 0.2)).astype(int)
            if floor.sum() > cfg.N:
                floor = None
        return greedy_allocate(g, cfg.N, floor)

    def act(self, obs):
        return self.allocate(self.loads(obs), obs)


# ------------------------------------------------- SDQ-Match transient core
class SDQTransient(Policy):
    """SDQ-Match matcher on the exact two-phase transient Erlang-A cost.

    For every spot it evaluates the expected waiting time accrued over the
    epoch's two phases (before / after repositioned aircraft arrive) as a
    function of the spot's net supply change d, and splits the idle fleet by
    marginal-cost water-filling on the convex envelope of those costs. The
    reposition latency enters the cost directly: an aircraft that leaves a spot
    is lost to it in phase 1, and an aircraft that arrives helps only in phase 2.

    ``scale`` (S x 2, optional) multiplies the phase loads per spot; it is the
    action of the learned (reinforcement-learning) layer.
    """
    name = "SDQ-core"
    uses_forecast = True

    def __init__(self, cfg, forecaster, gamma1=1.0, gamma2=1.0, H2=20,
                 lead=True, exact=True, inbound_exact=True, theta_online=False,
                 theta_prior=200.0):
        super().__init__(cfg, forecaster)
        from .transient import TransientCost
        self.gamma1, self.gamma2 = gamma1, gamma2
        self.lead, self.exact = lead, exact
        # inbound aircraft serve only from their landing time on (phase 1 is split there)
        self.inbound_exact = inbound_exact
        self.cmax = min(cfg.N, 40)
        self.tc = TransientCost(cfg.mu, cfg.theta, cfg.tau_rep, H2, cmax=self.cmax,
                                nmax=self.cmax + 45)
        self.H2 = H2
        # online abandonment rate: by Proposition 2, abandonments = theta * waiting time,
        # so theta_hat = (abandonments + k*theta0) / (waiting time + k), shrunk to theta0
        self.theta_online, self.theta_prior = theta_online, theta_prior
        self._tcs = {0: self.tc}

    def theta_hat(self) -> float:
        env = self.env
        t = env.t
        waited = float(env.q_hist[:t].sum())
        k = self.theta_prior
        return (env.n_abandon + k * self.cfg.theta) / (waited + k)

    def cost_model(self):
        """TransientCost for the current abandonment-rate estimate (log grid, cached)."""
        if not self.theta_online:
            return self.tc
        from .transient import TransientCost
        j = int(np.clip(np.round(8 * np.log2(max(self.theta_hat(), 1e-6) / self.cfg.theta)),
                        -24, 16))
        if j not in self._tcs:
            th = self.cfg.theta * 2.0 ** (j / 8)
            self._tcs[j] = TransientCost(self.cfg.mu, th, self.cfg.tau_rep, self.H2,
                                         cmax=self.cmax, nmax=self.cmax + 45)
        return self._tcs[j]

    def phase_loads(self, obs):
        cfg = self.cfg
        t = obs["t"]
        tau = cfg.tau_rep
        lam1 = self.fc.mean_rate(t, t + tau)
        if self.lead:
            lam2 = self.fc.mean_rate(t + tau, t + tau + self.H2)
        else:
            lam2 = lam1          # reactive: the present rate is assumed to persist
        a1 = self.gamma1 * lam1 * cfg.mean_service
        a2 = self.gamma2 * lam2 * cfg.mean_service
        return a1, a2

    def allocate(self, obs, a1, a2, resid=None):
        from .transient import convex_minorant_increments
        cfg = self.cfg
        idle, committed = obs["idle"], obs["committed"]
        n0 = obs["busy"] + obs["q"]
        D = int(idle.sum())
        if D == 0:
            return committed.copy()
        incs = []
        etas = obs.get("inbound_eta")
        tc = self.cost_model()
        for s in range(cfg.S):
            if self.inbound_exact and etas is not None and (etas[s] > 0).any():
                present = int(committed[s]) - len(etas[s])
                f = tc.delta_costs_inbound(int(n0[s]), present, etas[s], int(idle[s]),
                                           float(a1[s]), float(a2[s]), D)
            else:
                f = tc.delta_costs(int(n0[s]), int(committed[s]), int(idle[s]),
                                   float(a1[s]), float(a2[s]), D)
            if resid is not None:
                # learned convex residual in the phase-2 supply c2 = committed + d
                c2 = committed[s] + np.arange(-int(idle[s]), D + 1)
                f = f + resid[0][s] * c2 + max(0.0, resid[1][s]) * c2 ** 2
            incs.append(convex_minorant_increments(f) if self.exact else np.diff(f))
        d = -idle.copy()
        pos = np.zeros(cfg.S, dtype=int)
        for _ in range(D):
            best, bs = np.inf, -1
            for s in range(cfg.S):
                if pos[s] < len(incs[s]):
                    v = incs[s][pos[s]]
                    if v < best - 1e-12:
                        best, bs = v, s
            d[bs] += 1
            pos[bs] += 1
        return committed + d

    def act(self, obs):
        a1, a2 = self.phase_loads(obs)
        return self.allocate(obs, a1, a2)
