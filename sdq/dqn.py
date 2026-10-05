"""Deep Q-network matcher (learned baseline).

Following sequential per-aircraft dispatching, at every epoch the idle
aircraft are assigned one at a time: the Q-network reads the concatenated
spot features, a one-hot of the aircraft's current spot and the assignments
already made this epoch, and scores the S destinations (its own spot means
"stay"). Training is double DQN with experience replay and a target network
on the matching reward; the epoch reward is credited to the last
assignment of the epoch.
"""
from __future__ import annotations

import numpy as np

from .policies import Policy
from .rl import NF, NG, spot_features


def relu_forward(params, x):
    h = x
    n = len(params) // 2
    for i in range(n):
        h = h @ params[2 * i] + params[2 * i + 1]
        if i < n - 1:
            h = np.maximum(h, 0)
    return h


def dqn_input(f, g, S, cur, assigned, N):
    oh = np.zeros(S, dtype=np.float32)
    oh[cur] = 1.0
    return np.concatenate([f.ravel(), g, oh, assigned / max(1.0, N / S)]).astype(np.float32)


def dqn_in_dim(S):
    return S * NF + NG + 2 * S


class DQNPolicy(Policy):
    name = "DQN"
    uses_forecast = True

    def __init__(self, cfg, forecaster, qnet=None, eps=0.0, seed=0):
        super().__init__(cfg, forecaster)
        self.qnet, self.eps = qnet, eps
        self.rng = np.random.default_rng(seed)
        self.micro = []          # (x, a) of the current epoch, for training

    def act(self, obs):
        cfg = self.cfg
        S, N = cfg.S, cfg.N
        f, g = spot_features(obs, self.fc, cfg)
        committed = obs["committed"].copy()
        assigned = np.zeros(S, dtype=np.float32)
        self.micro = []
        idle_spots = np.repeat(np.arange(S), obs["idle"])
        for cur in idle_spots:
            x = dqn_input(f, g, S, cur, assigned, N)
            if self.rng.random() < self.eps:
                a = int(self.rng.integers(S))
            else:
                a = int(np.argmax(relu_forward(self.qnet, x[None])[0]))
            self.micro.append((x, a))
            if a != cur:
                committed[cur] -= 1
                committed[a] += 1
                assigned[a] += 1
        return committed
