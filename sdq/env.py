"""Discrete-event simulator of a low-altitude sightseeing operation.

S boarding spots share N aircraft. Tourists arrive at each spot, queue FCFS,
and abandon when their wait exceeds their patience. An aircraft at a spot
serves the head-of-line tourist (one sightseeing loop) and is then free again
at the same spot. Every ``kappa`` steps a matching policy returns a target
allocation c* (sum N); idle aircraft at surplus spots are sent to deficit
spots and become available after ``tau_rep`` steps.

All randomness of a simulated day (arrivals, patience, flight durations, day
multiplier, peak timing) is drawn up front from the seed, so every policy
sees exactly the same day: common random numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from .demand import rate_matrix, day_parameters

IDLE, BUSY, INBOUND = 0, 1, 2


@dataclass
class EnvConfig:
    S: int = 5
    N: int = 10
    T: int = 720
    lam0: float = 0.16
    mean_service: float = 7.0
    mean_patience: float = 45.0
    tau_rep: int = 5
    kappa: int = 5
    w_bar: float = 15.0
    w0: float = 20.0
    profile: str = "weekday"
    amp: float = 2.0
    base: float = 0.35
    width_frac: float = 0.08
    day_sigma: float = 0.10
    peak_jitter: float = 20.0
    arrival: str = "poisson"          # poisson | negbin (over-dispersed)
    nb_size: float = 2.0              # negative-binomial size parameter
    service_dist: str = "geometric"   # geometric | lognormal
    patience_dist: str = "geometric"  # geometric | uniform
    # reward weights of the matching MDP, r = b1*served - b2*sum_q - b3*abandoned
    beta1: float = 1.0
    beta2: float = 0.05
    beta3: float = 1.0

    def to_dict(self):
        return asdict(self)

    @property
    def mu(self) -> float:
        return 1.0 / self.mean_service

    @property
    def theta(self) -> float:
        return 1.0 / self.mean_patience


def _discrete(rng, dist: str, mean: float, size: int) -> np.ndarray:
    if dist == "geometric":
        # memoryless on {1,2,...} with the given mean
        return rng.geometric(1.0 / mean, size=size)
    if dist == "lognormal":
        sig = 0.8
        x = rng.lognormal(np.log(mean) - 0.5 * sig ** 2, sig, size=size)
        return np.maximum(1, np.rint(x)).astype(int)
    if dist == "uniform":
        return rng.integers(1, int(2 * mean), size=size)
    raise ValueError(dist)


@dataclass
class DayRealisation:
    rates: np.ndarray                  # (T, S) true arrival rates
    arrivals: list                     # per spot: arrays (t_arr, patience, service)
    n_arrivals: int
    day_mult: float
    peak_shift: np.ndarray = field(default_factory=lambda: np.zeros(2))


def realise_day(cfg: EnvConfig, seed: int) -> DayRealisation:
    rng = np.random.default_rng(seed)
    day_mult, peak_shift = day_parameters(rng, cfg.day_sigma, cfg.peak_jitter)
    rates = rate_matrix(cfg.S, cfg.T, cfg.lam0, cfg.profile, cfg.base, cfg.amp,
                        cfg.width_frac, peak_shift, day_mult)
    if cfg.arrival == "poisson":
        counts = rng.poisson(rates)
    elif cfg.arrival == "negbin":
        r = cfg.nb_size
        p = r / (r + rates)
        counts = rng.negative_binomial(r, p)
    else:
        raise ValueError(cfg.arrival)
    arrivals = []
    total = 0
    for s in range(cfg.S):
        t_arr = np.repeat(np.arange(cfg.T), counts[:, s])
        n = len(t_arr)
        pat = _discrete(rng, cfg.patience_dist, cfg.mean_patience, n)
        svc = _discrete(rng, cfg.service_dist, cfg.mean_service, n)
        arrivals.append((t_arr, pat, svc))
        total += n
    return DayRealisation(rates, arrivals, total, day_mult, peak_shift)


class ScenicEnv:
    """Gym-style environment. ``reset(seed)`` -> obs; ``step(target)`` advances one epoch."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int, day: DayRealisation | None = None):
        cfg = self.cfg
        self.day = day if day is not None else realise_day(cfg, seed)
        self.t = 0
        S, N = cfg.S, cfg.N
        self.loc = np.arange(N) % S             # identical initial positions
        self.status = np.zeros(N, dtype=int)
        self.t_free = np.zeros(N, dtype=int)
        self.queues = [[] for _ in range(S)]    # entries: [t_arr, deadline, service]
        self._ptr = [0] * S
        self.waits = []                         # waits of served tourists
        self.wait_spot = []
        self.n_served = 0
        self.n_abandon = 0
        self.arr_hist = np.zeros((cfg.T, S), dtype=int)
        self.q_hist = np.zeros((cfg.T, S), dtype=int)
        self.n_moves = 0
        return self.observe()

    # ------------------------------------------------------------- observation
    def counts(self):
        S = self.cfg.S
        idle = np.bincount(self.loc[self.status == IDLE], minlength=S)
        busy = np.bincount(self.loc[self.status == BUSY], minlength=S)
        inbound = np.bincount(self.loc[self.status == INBOUND], minlength=S)
        return idle, busy, inbound

    def observe(self):
        idle, busy, inbound = self.counts()
        q = np.array([len(x) for x in self.queues])
        t = self.t
        lo = max(0, t - 15)
        recent = self.arr_hist[lo:t].sum(0) / max(1, t - lo)
        # remaining flight time of busy aircraft, per spot
        rem = np.zeros(self.cfg.S)
        m = self.status == BUSY
        np.add.at(rem, self.loc[m], np.maximum(0, self.t_free[m] - t))
        # oldest wait in each queue
        head = np.array([t - x[0][0] if x else 0 for x in self.queues], dtype=float)
        # steps until each inbound aircraft lands, per spot
        m = self.status == INBOUND
        eta = [np.maximum(0, self.t_free[m & (self.loc == s)] - t) for s in range(self.cfg.S)]
        return dict(t=t, q=q, idle=idle, busy=busy, inbound=inbound,
                    committed=idle + busy + inbound, recent_rate=recent,
                    busy_remaining=rem, head_wait=head, inbound_eta=eta)

    # -------------------------------------------------------------------- step
    def apply_target(self, target):
        """Move idle aircraft from surplus to deficit spots (idle units only)."""
        cfg = self.cfg
        target = np.asarray(target, dtype=int)
        assert target.sum() == cfg.N and (target >= 0).all(), target
        idle, busy, inbound = self.counts()
        committed = idle + busy + inbound
        deficit = target - committed
        if (deficit > 0).sum() == 0:
            return 0
        moves = 0
        # sources: surplus spots with idle aircraft
        donors = [s for s in np.argsort(deficit) if deficit[s] < 0 and idle[s] > 0]
        for s in donors:
            give = min(idle[s], -deficit[s])
            ids = np.flatnonzero((self.loc == s) & (self.status == IDLE))[:give]
            for j in ids:
                d = int(np.argmax(deficit))
                if deficit[d] <= 0:
                    break
                self.loc[j] = d
                self.status[j] = INBOUND
                self.t_free[j] = self.t + cfg.tau_rep
                deficit[d] -= 1
                deficit[s] += 1
                moves += 1
        self.n_moves += moves
        return moves

    def _sim_step(self):
        cfg, t = self.cfg, self.t
        # 1. completions and reposition arrivals
        done = (self.status != IDLE) & (self.t_free <= t)
        self.status[done] = IDLE
        served = abandoned = 0
        for s in range(cfg.S):
            t_arr, pat, svc = self.day.arrivals[s]
            # 2. arrivals of this step
            p = self._ptr[s]
            q = self.queues[s]
            while p < len(t_arr) and t_arr[p] == t:
                q.append((t, t + pat[p], svc[p]))
                p += 1
            self.arr_hist[t, s] = p - self._ptr[s]
            self._ptr[s] = p
            # 3. abandonment: a tourist waiting longer than the patience leaves
            if q:
                keep = [x for x in q if x[1] >= t]
                abandoned += len(q) - len(keep)
                q = keep
            # 4. service, FCFS, by idle aircraft at the spot
            if q:
                free = np.flatnonzero((self.loc == s) & (self.status == IDLE))
                k = min(len(free), len(q))
                for i in range(k):
                    ta, _, sv = q[i]
                    j = free[i]
                    self.status[j] = BUSY
                    self.t_free[j] = t + sv
                    w = t - ta
                    self.waits.append(w)
                    self.wait_spot.append(s)
                q = q[k:]
                served += k
            self.queues[s] = q
            self.q_hist[t, s] = len(q)
        self.n_served += served
        self.n_abandon += abandoned
        self.t += 1
        return served, abandoned, int(self.q_hist[t].sum())

    def step(self, target):
        """Apply the target allocation, simulate kappa steps, return (obs, reward, done, info)."""
        cfg = self.cfg
        self.apply_target(target)
        r = 0.0
        srv = ab = qsum = 0
        for _ in range(cfg.kappa):
            if self.t >= cfg.T:
                break
            s_, a_, q_ = self._sim_step()
            srv += s_
            ab += a_
            qsum += q_
            r += cfg.beta1 * s_ - cfg.beta2 * q_ - cfg.beta3 * a_
        done = self.t >= cfg.T
        info = dict(served=srv, abandoned=ab, qsum=qsum)
        return self.observe(), r, done, info

    # ----------------------------------------------------------------- metrics
    def metrics(self):
        cfg = self.cfg
        K = max(1, self.day.n_arrivals)
        waits = np.asarray(self.waits, dtype=float)
        n_srv = len(waits)
        sdmr = float((waits <= cfg.w_bar).sum()) / K
        aqw = float(waits.mean()) if n_srv else float(cfg.T)
        ab = 1.0 - n_srv / K
        qass = sdmr * np.exp(-aqw / cfg.w0) * (1.0 - ab)
        p95 = float(np.percentile(waits, 95)) if n_srv else float(cfg.T)
        # cross-spot fairness: coefficient of variation of the per-spot mean wait
        ws = np.asarray(self.wait_spot)
        per = np.array([waits[ws == s].mean() if (ws == s).any() else 0.0
                        for s in range(cfg.S)])
        cv = float(per.std() / per.mean()) if per.mean() > 0 else 0.0
        return dict(QASS=100 * qass, SDMR=100 * sdmr, AQW=aqw, P95=p95,
                    AB=100 * ab, CV=cv, arrivals=K, served=n_srv,
                    abandoned=self.n_abandon, moves=self.n_moves)


def run_episode(env: ScenicEnv, policy, seed: int, day: DayRealisation | None = None):
    obs = env.reset(seed, day)
    policy.reset(env)
    done = False
    while not done:
        target = policy.act(obs)
        obs, _, done, _ = env.step(target)
    return env.metrics()
