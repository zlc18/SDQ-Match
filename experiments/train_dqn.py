"""Double-DQN training of the deep Q-network matcher baseline.

Usage: python experiments/train_dqn.py --out results/models/dqn_main.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.env import ScenicEnv  # noqa: E402
from sdq.forecast import Forecaster  # noqa: E402
from sdq.dqn import DQNPolicy, dqn_in_dim  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="main")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma_d", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--updates_per_epoch", type=int, default=2)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--val_seeds", default="100-119")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = get_scenario(args.scenario)
    S = cfg.S
    D = dqn_in_dim(S)
    H = args.hidden

    def net():
        return nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(),
                             nn.Linear(H, S))

    q, qt = net(), net()
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=args.lr)

    def np_params(m):
        out = []
        for l in m:
            if isinstance(l, nn.Linear):
                out += [l.weight.detach().numpy().T.copy(), l.bias.detach().numpy().copy()]
        return out

    cap = 200_000
    X = np.zeros((cap, D), np.float32)
    A = np.zeros(cap, np.int64)
    R = np.zeros(cap, np.float32)
    X2 = np.zeros((cap, D), np.float32)
    G = np.zeros(cap, np.float32)      # discount to next state (0 if terminal)
    ptr = size = 0

    def push(x, a, r, x2, g):
        nonlocal ptr, size
        X[ptr], A[ptr], R[ptr], G[ptr] = x, a, r, g
        X2[ptr] = x2 if x2 is not None else 0
        ptr = (ptr + 1) % cap
        size = min(size + 1, cap)

    def update():
        idx = rng.integers(0, size, args.batch)
        x, a, r = torch.tensor(X[idx]), torch.tensor(A[idx]), torch.tensor(R[idx])
        x2, g = torch.tensor(X2[idx]), torch.tensor(G[idx])
        with torch.no_grad():
            a2 = q(x2).argmax(1, keepdim=True)
            tgt = r + g * qt(x2).gather(1, a2).squeeze(1)
        pred = q(x).gather(1, a[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()

    def evaluate(params):
        env = ScenicEnv(cfg)
        lo, hi = map(int, args.val_seeds.split("-"))
        out = []
        for sd in range(lo, hi + 1):
            pol = DQNPolicy(cfg, Forecaster(cfg), params, eps=0.0)
            obs = env.reset(sd)
            pol.reset(env, sd)
            done = False
            while not done:
                obs, _, done, _ = env.step(pol.act(obs))
            out.append(env.metrics()["QASS"])
        return float(np.mean(out))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    env = ScenicEnv(cfg)
    best, log, n_upd = -np.inf, [], 0
    t0 = time.time()
    for ep in range(args.episodes + 1):
        if ep % args.eval_every == 0:
            params = np_params(q)
            vq = evaluate(params)
            log.append(dict(episode=ep, val_qass=vq, time=time.time() - t0))
            print(f"[{ep}] val QASS {vq:.2f} t={time.time()-t0:.0f}s", flush=True)
            if vq > best:
                best = vq
                np.savez(args.out, *params, val_qass=vq, episode=ep)
        if ep == args.episodes:
            break
        eps = max(0.05, 1.0 - ep / (0.5 * args.episodes))
        pol = DQNPolicy(cfg, Forecaster(cfg), np_params(q), eps=eps, seed=ep)
        obs = env.reset(30_000 + ep)
        pol.reset(env, 30_000 + ep)
        pending = None          # last micro-transition awaiting its successor
        acc = 0.0
        done = False
        while not done:
            pol.qnet = np_params(q) if n_upd % 50 == 0 else pol.qnet
            target = pol.act(obs)
            micro = pol.micro
            if micro:
                if pending is not None:
                    push(pending[0], pending[1], acc, micro[0][0], args.gamma_d)
                    acc = 0.0
                for k in range(len(micro) - 1):
                    push(micro[k][0], micro[k][1], 0.0, micro[k + 1][0], 1.0)
                pending = micro[-1]
            obs, r, done, _ = env.step(target)
            acc += r / 10.0
            if size > 1000:
                for _ in range(args.updates_per_epoch):
                    update()
                    n_upd += 1
                    if n_upd % 500 == 0:
                        qt.load_state_dict(q.state_dict())
        if pending is not None:
            push(pending[0], pending[1], acc, None, 0.0)
    with open(args.out.replace(".npz", "_log.json"), "w") as f:
        json.dump(dict(args=vars(args), log=log, best_val=best), f, indent=1)
    print("best val QASS", best)


if __name__ == "__main__":
    main()
