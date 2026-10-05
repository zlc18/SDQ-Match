"""Learned load correction of SDQ-Match and the end-to-end PPO baseline.

SDQ-Match: a per-spot network, shared across spots (so it is independent of S),
reads local queue/fleet/forecast features and outputs delta_s in R^2, the
log-scale corrections of the two phase loads of the transient Erlang-A matcher.
The allocation itself is always produced by the exact water-filling step, so the
network acts in the space of queueing loads and never in the combinatorial space
of allocations. The network is trained by CRN-paired evolution strategies on the
matching reward (experiments/train_es.py); a PPO trainer for the same network is
also provided (experiments/train_rl.py --kind sdq).

PPO-E2E baseline: a fixed-width actor reads the concatenated features of all
spots and outputs the logits of the fleet split directly (largest remainder).
"""
from __future__ import annotations

import numpy as np

from .env import ScenicEnv, EnvConfig
from .forecast import Forecaster
from .policies import Policy, SDQTransient, largest_remainder

NF = 14   # per-spot features (version 1)
NF2 = 18  # per-spot features (version 2: adds observed rate, forecast residual, latency, test state)
NG = 6    # global features


def n_spot_features(version: int) -> int:
    return NF2 if version == 2 else NF


def spot_features(obs, fc: Forecaster, cfg: EnvConfig, version: int = 1):
    t = obs["t"]
    tau = cfg.tau_rep
    Ts = cfg.mean_service
    scale = max(1.0, cfg.N / cfg.S)
    l0 = fc.mean_rate(t, t + tau) * Ts
    l1 = fc.mean_rate(t + tau, t + tau + 20) * Ts
    l2 = fc.mean_rate(t + tau + 20, t + tau + 60) * Ts
    busy = obs["busy"]
    rem = obs["busy_remaining"] / np.maximum(1, busy) / Ts
    ph = 2 * np.pi * t / cfg.T
    f = np.stack([
        obs["q"] / 10.0, obs["idle"] / scale, busy / scale, obs["inbound"] / scale,
        obs["committed"] / scale, l0 / scale, l1 / scale, l2 / scale,
        (l1 - l0) / scale, obs["head_wait"] / cfg.mean_patience, rem,
        np.full(cfg.S, np.sin(ph)), np.full(cfg.S, np.cos(ph)),
        (obs["q"] + busy) / np.maximum(1, obs["committed"]) / 5.0,
    ], 1).astype(np.float32)
    if version == 2:
        obs_rate = obs["recent_rate"] * Ts
        switched = float(getattr(fc, "switched", False))
        f = np.concatenate([f, np.stack([
            obs_rate / scale, (obs_rate - l0) / scale, np.full(cfg.S, tau / 10.0),
            np.full(cfg.S, switched)], 1).astype(np.float32)], 1)
    g = np.array([obs["q"].sum() / cfg.N, obs["idle"].sum() / cfg.N,
                  l0.sum() / cfg.N, l1.sum() / cfg.N, np.sin(ph), np.cos(ph)],
                 dtype=np.float32)
    return f, g


# ----------------------------------------------------------- numpy MLP
def mlp_forward(params, x):
    h = x
    n = len(params) // 2
    for i in range(n):
        W, b = params[2 * i], params[2 * i + 1]
        h = h @ W + b
        if i < n - 1:
            h = np.tanh(h)
    return h


class SDQMatch(Policy):
    """SDQ-Match: transient Erlang-A water-filling + a learned component.

    ``place`` selects where the learned component acts:
      load  - per-spot log-scale corrections of the two phase loads (2 outputs/spot)
      meta  - a global meta-controller rescaling the phase loads of all spots (2 outputs)
      resid - a per-spot learned convex residual cost a_s*c2 + b_s*c2^2, b_s >= 0,
              added to the transient cost, so water-filling stays exact (2 outputs/spot)
    """
    name = "SDQ-Match"
    uses_forecast = True

    def __init__(self, cfg, forecaster, actor=None, log_std=None, stochastic=False,
                 core_kw=None, seed=0, place="load", feat=1, train_tau=None):
        super().__init__(cfg, forecaster)
        self.core = SDQTransient(cfg, forecaster, **(core_kw or {}))
        self.feat = feat
        # the learned correction is not extrapolated to reposition latencies it was not
        # trained on; there the matcher falls back to the queueing core
        if train_tau is not None and cfg.tau_rep != train_tau:
            actor = None
        self.actor = actor if (train_tau is None or cfg.tau_rep == train_tau) else None
        self.log_std = log_std
        self.stochastic = stochastic
        self.place = place
        self.rng = np.random.default_rng(seed)
        self.trace = []

    def reset(self, env, seed: int = 0):
        super().reset(env, seed)
        self.core.env = env
        self.trace = []

    def _action(self, x, g):
        if isinstance(self.actor, tuple):      # ensemble: average the member outputs
            mean = np.mean([mlp_forward(m, x) for m in self.actor], axis=0)
        else:
            mean = mlp_forward(self.actor, x)
        if self.stochastic:
            std = np.exp(self.log_std)
            act = mean + std * self.rng.standard_normal(mean.shape)
            logp = (-0.5 * ((act - mean) / std) ** 2 - self.log_std
                    - 0.5 * np.log(2 * np.pi)).sum()
        else:
            act, logp = mean, 0.0
        act = np.clip(act, -2.0, 2.0)
        self.trace.append((x, g, act.astype(np.float32), logp))
        return act

    def act(self, obs):
        a1, a2 = self.core.phase_loads(obs)
        resid = None
        if self.actor is not None:
            f, g = spot_features(obs, self.fc, self.cfg, self.feat)
            if self.place == "meta":
                x = np.concatenate([f.mean(0), f.max(0), g])[None]
                act = self._action(x, g)[0]
                a1 = a1 * np.exp(act[0])
                a2 = a2 * np.exp(act[1])
            else:
                x = np.concatenate([f, np.repeat(g[None], self.cfg.S, 0)], 1)
                act = self._action(x, g)
                if self.place == "load":
                    a1 = a1 * np.exp(act[:, 0])
                    a2 = a2 * np.exp(act[:, 1])
                else:
                    resid = (act[:, 0], np.log1p(np.exp(act[:, 1])) - np.log(2.0))
        return self.core.allocate(obs, a1, a2, resid)


class PPOE2E(Policy):
    """End-to-end deep-RL matcher: global MLP -> fleet split logits."""
    name = "PPO"
    uses_forecast = True

    def __init__(self, cfg, forecaster, actor=None, log_std=None, stochastic=False, seed=0):
        super().__init__(cfg, forecaster)
        self.actor, self.log_std, self.stochastic = actor, log_std, stochastic
        self.rng = np.random.default_rng(seed)
        self.trace = []

    def reset(self, env, seed: int = 0):
        super().reset(env, seed)
        self.trace = []

    def act(self, obs):
        f, g = spot_features(obs, self.fc, self.cfg)
        x = np.concatenate([f.ravel(), g])[None]
        mean = mlp_forward(self.actor, x)[0]
        if self.stochastic:
            std = np.exp(self.log_std)
            act = mean + std * self.rng.standard_normal(mean.shape)
            logp = (-0.5 * ((act - mean) / std) ** 2 - self.log_std
                    - 0.5 * np.log(2 * np.pi)).sum()
        else:
            act, logp = mean, 0.0
        act = np.clip(act, -5, 5)
        self.trace.append((x[0], g, act.astype(np.float32), logp))
        w = np.exp(act - act.max())
        return largest_remainder(w, self.cfg.N)


# ------------------------------------------------------------- rollouts
def rollout(args):
    """Worker: run one episode with a stochastic policy, return the trajectory."""
    kind, cfg_dict, actor, log_std, seed, core_kw = args
    cfg = EnvConfig(**cfg_dict)
    env = ScenicEnv(cfg)
    fc = Forecaster(cfg)
    if kind == "sdq":
        pol = SDQMatch(cfg, fc, actor, log_std, stochastic=True, core_kw=core_kw, seed=seed)
    else:
        pol = PPOE2E(cfg, fc, actor, log_std, stochastic=True, seed=seed)
    obs = env.reset(seed)
    pol.reset(env, seed)
    rews, done = [], False
    while not done:
        target = pol.act(obs)
        obs, r, done, _ = env.step(target)
        rews.append(r)
    X = np.stack([tr[0] for tr in pol.trace])
    G = np.stack([tr[1] for tr in pol.trace])
    A = np.stack([tr[2] for tr in pol.trace])
    LP = np.array([tr[3] for tr in pol.trace], dtype=np.float32)
    return dict(X=X, G=G, A=A, LP=LP, R=np.array(rews, dtype=np.float32),
                metrics=env.metrics())
