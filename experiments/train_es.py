"""CRN-paired antithetic evolution strategies for the learned layer of SDQ-Match.

Each iteration draws P perturbations eps_i and P training days d_i. The two
mirrored policies theta +/- sigma*eps_i are run on the *same* day d_i (common
random numbers), so the day-to-day demand variation cancels in the return
difference R(theta+sigma eps_i, d_i) - R(theta-sigma eps_i, d_i). The search
gradient uses rank-based fitness shaping and Adam.

Usage:
  python experiments/train_es.py --place load  --out results/models/es_load.npz
  python experiments/train_es.py --place meta  --out results/models/es_meta.npz
  python experiments/train_es.py --place resid --out results/models/es_resid.npz
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
from sdq.rl import NG, SDQMatch, n_spot_features  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402


def shapes_for(place, hidden, feat=1):
    nf = n_spot_features(feat)
    if place == "meta":
        sizes = [2 * nf + NG, hidden, hidden, 2]
    else:
        sizes = [nf + NG, hidden, hidden, 2]
    out = []
    for i in range(len(sizes) - 1):
        out += [(sizes[i], sizes[i + 1]), (sizes[i + 1],)]
    return out


def unflatten(theta, shapes):
    out, k = [], 0
    for sh in shapes:
        n = int(np.prod(sh))
        out.append(theta[k:k + n].reshape(sh))
        k += n
    return out


def init_theta(shapes, rng):
    parts = []
    for i, sh in enumerate(shapes):
        if len(sh) == 2 and i < len(shapes) - 2:
            parts.append(rng.normal(0, 1 / np.sqrt(sh[0]), size=sh).ravel())
        else:
            parts.append(np.zeros(int(np.prod(sh))))   # biases and last layer: zero
    return np.concatenate(parts)


def episode(args):
    place, cfgd, actor, seed, core_kw, objective = args[:6]
    feat = args[6] if len(args) > 6 else 1
    cfgd = dict(cfgd)
    hist_profile = cfgd.pop("_hist_profile", None)
    noise = cfgd.pop("_noise", 0.0)
    cfg = EnvConfig(**cfgd)
    hist_cfg = EnvConfig(**{**cfgd, "profile": hist_profile}) if hist_profile else None
    env = ScenicEnv(cfg)
    pol = SDQMatch(cfg, Forecaster(cfg, noise=noise, hist_cfg=hist_cfg), actor,
                   core_kw=core_kw, place=place, feat=feat)
    obs = env.reset(seed)
    pol.reset(env, seed)
    R, done = 0.0, False
    while not done:
        obs, r, done, _ = env.step(pol.act(obs))
        R += r
    m = env.metrics()
    return (m["QASS"] if objective == "qass" else R), m["QASS"]


DR_GRID = dict(lam0=[0.12, 0.14, 0.16, 0.18, 0.20], tau_rep=[2, 3, 5, 8, 10],
               mean_patience=[20.0, 30.0, 45.0, 60.0], profile=["weekday", "weekend", "festival"])


DR_GRID_V2 = dict(DR_GRID, _hist_profile=[None, None, "weekday", "weekend", "festival"],
                  _noise=[0.0, 0.0, 0.1, 0.3, 0.5])


def sample_over(rng, grid=None):
    """Domain randomisation of the operating parameters for one training day."""
    grid = grid or DR_GRID
    return {k: v[int(rng.integers(len(v)))] for k, v in grid.items()}


def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x)] = np.arange(len(x))
    return r / (len(x) - 1) - 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", choices=["load", "meta", "resid"], default="load")
    ap.add_argument("--scenario", default="main")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--objective", choices=["return", "qass"], default="return")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--val_seeds", default="100-119")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--core_kw", default="{}")
    ap.add_argument("--randomize", action="store_true",
                    help="domain randomisation of load, latency, patience and profile on "
                         "training days; validation then averages the calibration days and "
                         "randomised validation days")
    ap.add_argument("--feat", type=int, default=1, help="feature version of the actor")
    ap.add_argument("--grid", default="v1", choices=["v1", "v2"],
                    help="v2 also randomises the forecast history profile and forecast noise")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = get_scenario(args.scenario)
    cfgd = cfg.to_dict()
    core_kw = json.loads(args.core_kw)
    shapes = shapes_for(args.place, args.hidden, args.feat)
    grid = DR_GRID_V2 if args.grid == "v2" else DR_GRID
    theta = init_theta(shapes, rng)
    m_adam = np.zeros_like(theta)
    v_adam = np.zeros_like(theta)
    lo, hi = map(int, args.val_seeds.split("-"))
    val_seeds = list(range(lo, hi + 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log, best = [], -np.inf
    pool = Pool(args.workers)
    t0 = time.time()
    day = 20_000 + 100_000 * args.seed
    for it in range(args.iters + 1):
        if it % args.eval_every == 0:
            actor = unflatten(theta, shapes)
            res = pool.map(episode, [(args.place, cfgd, actor, sd, core_kw, "qass", args.feat)
                                     for sd in val_seeds])
            vq = float(np.mean([r[1] for r in res]))
            entry = dict(iter=it, val_qass=vq, time=time.time() - t0)
            if args.randomize:
                vrng = np.random.default_rng(777)
                vjobs = [(args.place, {**cfgd, **sample_over(vrng, grid)}, actor, sd, core_kw,
                          "qass", args.feat) for sd in val_seeds]
                vr = float(np.mean([r[1] for r in pool.map(episode, vjobs)]))
                entry.update(val_calib=vq, val_random=vr)
                vq = 0.5 * (vq + vr)
                entry["val_qass"] = vq
            log.append(entry)
            print(f"[{it}] val QASS {vq:.2f} t={time.time()-t0:.0f}s", flush=True)
            if vq > best:
                best = vq
                np.savez(args.out, *actor, val_qass=vq, iter=it, place=args.place,
                         scenario=args.scenario, core_kw=json.dumps(core_kw), feat=args.feat)
        if it == args.iters:
            break
        eps = rng.standard_normal((args.pairs, len(theta)))
        jobs = []
        for i in range(args.pairs):
            cd = {**cfgd, **sample_over(rng, grid)} if args.randomize else cfgd
            for sgn in (1, -1):
                jobs.append((args.place, cd, unflatten(theta + sgn * args.sigma * eps[i], shapes),
                             day + i, core_kw, args.objective, args.feat))
        day += args.pairs
        res = np.array([r[0] for r in pool.map(episode, jobs)]).reshape(args.pairs, 2)
        diff = res[:, 0] - res[:, 1]
        if args.randomize:   # returns differ in scale across operating conditions
            diff = diff / (np.abs(res).mean(1) + 1.0)
        w = ranks(diff)
        grad = (w[:, None] * eps).sum(0) / (args.pairs * args.sigma)
        # Adam ascent
        b1, b2 = 0.9, 0.999
        m_adam = b1 * m_adam + (1 - b1) * grad
        v_adam = b2 * v_adam + (1 - b2) * grad ** 2
        mh = m_adam / (1 - b1 ** (it + 1))
        vh = v_adam / (1 - b2 ** (it + 1))
        theta = theta + args.lr * mh / (np.sqrt(vh) + 1e-8)
        log[-1].setdefault("pair_gap", []).append(float(np.abs(diff).mean()))
    pool.close()
    with open(args.out.replace(".npz", "_log.json"), "w") as f:
        json.dump(dict(args=vars(args), log=log, best_val=best), f, indent=1)
    print("best val QASS", best)


if __name__ == "__main__":
    main()
