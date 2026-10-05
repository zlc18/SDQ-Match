"""PPO training of the SDQ-Match learned layer, or of the end-to-end PPO baseline.

Usage:
  python experiments/train_rl.py --kind sdq --out results/models/sdq_main.npz
  python experiments/train_rl.py --kind e2e --out results/models/ppo_main.npz

Training days use seeds 10000+ (disjoint from validation 100-119 and
evaluation 200-229). The policy with the best mean validation QASS
(deterministic actions, validation seeds 100-119) is saved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.env import EnvConfig, ScenicEnv  # noqa: E402
from sdq.forecast import Forecaster  # noqa: E402
from sdq.rl import NF, NG, SDQMatch, PPOE2E, rollout  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402


def mlp(sizes, out_zero=False):
    import torch.nn as nn
    layers = []
    for i in range(len(sizes) - 1):
        lin = nn.Linear(sizes[i], sizes[i + 1])
        if i == len(sizes) - 2 and out_zero:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        layers.append(lin)
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def to_numpy(model):
    import torch.nn as nn
    out = []
    for m in model:
        if isinstance(m, nn.Linear):
            out += [m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy()]
    return out


def evaluate(kind, cfg, actor, seeds, core_kw):
    env = ScenicEnv(cfg)
    q = []
    for sd in seeds:
        fc = Forecaster(cfg)
        pol = (SDQMatch(cfg, fc, actor, core_kw=core_kw) if kind == "sdq"
               else PPOE2E(cfg, fc, actor))
        obs = env.reset(sd)
        pol.reset(env, sd)
        done = False
        while not done:
            obs, _, done, _ = env.step(pol.act(obs))
        q.append(env.metrics()["QASS"])
    return float(np.mean(q))


def _eval_job(a):
    kind, cfgd, actor, sd, core_kw = a
    return evaluate(kind, EnvConfig(**cfgd), actor, [sd], core_kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["sdq", "e2e"], default="sdq")
    ap.add_argument("--scenario", default="main")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--log_std0", type=float, default=-1.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--gamma_d", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--val_seeds", default="100-119")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--core_kw", default="{}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = get_scenario(args.scenario)
    core_kw = json.loads(args.core_kw)
    S = cfg.S
    lo, hi = map(int, args.val_seeds.split("-"))
    val_seeds = list(range(lo, hi + 1))

    if args.kind == "sdq":
        in_a, out_a = NF + NG, 2
        in_c = 2 * (NF + NG)
    else:
        in_a, out_a = S * NF + NG, S
        in_c = S * NF + NG
    H = args.hidden
    actor = mlp([in_a, H, H, out_a], out_zero=True)
    critic = mlp([in_c, H, H, 1])
    log_std = nn.Parameter(torch.full((out_a,), args.log_std0))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std],
                           lr=args.lr)

    def critic_in(X):
        if args.kind == "sdq":   # X: (B, S, F) -> pooled over spots
            return torch.cat([X.mean(1), X.max(1).values], -1)
        return X

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log = []
    best = -np.inf
    pool = Pool(args.workers)
    cfgd = cfg.to_dict()
    train_seed = 10_000 + 1000 * args.seed
    t0 = time.time()
    for it in range(args.iters + 1):
        if it % args.eval_every == 0:
            np_actor = to_numpy(actor)
            vq = float(np.mean(pool.map(_eval_job, [(args.kind, cfgd, np_actor, sd, core_kw)
                                                     for sd in val_seeds])))
            log.append(dict(iter=it, val_qass=vq, time=time.time() - t0,
                            log_std=log_std.detach().numpy().tolist()))
            print(f"[{it}] val QASS {vq:.2f}  t={time.time()-t0:.0f}s", flush=True)
            if vq > best:
                best = vq
                np.savez(args.out, *np_actor, val_qass=vq, iter=it, kind=args.kind,
                         scenario=args.scenario, core_kw=json.dumps(core_kw))
        if it == args.iters:
            break
        np_actor = to_numpy(actor)
        ls = log_std.detach().numpy().copy()
        jobs = [(args.kind, cfgd, np_actor, ls, train_seed + it * args.episodes + e, core_kw)
                for e in range(args.episodes)]
        trajs = pool.map(rollout, jobs)
        # ---- GAE per episode
        Xs, As, LPs, ADV, RET = [], [], [], [], []
        with torch.no_grad():
            for tr in trajs:
                X = torch.tensor(tr["X"])
                v = critic(critic_in(X)).squeeze(-1).numpy()
                r = tr["R"] / 10.0
                T = len(r)
                adv = np.zeros(T, dtype=np.float32)
                gae = 0.0
                for k in reversed(range(T)):
                    nv = v[k + 1] if k + 1 < T else 0.0
                    delta = r[k] + args.gamma_d * nv - v[k]
                    gae = delta + args.gamma_d * args.lam * gae
                    adv[k] = gae
                Xs.append(tr["X"]); As.append(tr["A"]); LPs.append(tr["LP"])
                ADV.append(adv); RET.append(adv + v)
        X = torch.tensor(np.concatenate(Xs))
        A = torch.tensor(np.concatenate(As))
        LP0 = torch.tensor(np.concatenate(LPs))
        adv = torch.tensor(np.concatenate(ADV))
        ret = torch.tensor(np.concatenate(RET))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(X)
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 256):
                idx = perm[i:i + 256]
                mean = actor(X[idx])
                std = torch.exp(log_std)
                lp = (-0.5 * ((A[idx] - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi))
                lp = lp.reshape(len(idx), -1).sum(-1)
                ratio = torch.exp(lp - LP0[idx])
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv[idx]
                v = critic(critic_in(X[idx])).squeeze(-1)
                loss = -torch.min(s1, s2).mean() + 0.5 * ((v - ret[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()
        mq = np.mean([tr["metrics"]["QASS"] for tr in trajs])
        print(f"  it {it} train QASS {mq:.2f} ret {np.mean([tr['R'].sum() for tr in trajs]):.1f}",
              flush=True)
        log[-1].setdefault("train", []).append(float(mq))
    pool.close()
    with open(args.out.replace(".npz", "_log.json"), "w") as f:
        json.dump(dict(args=vars(args), log=log, best_val=best), f, indent=1)
    print("best val QASS", best)


if __name__ == "__main__":
    main()
