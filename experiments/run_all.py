"""Reproduce every experiment of the paper.

  python experiments/run_all.py --stage tune      # hyperparameter selection (validation seeds)
  python experiments/run_all.py --stage train     # learned layer, DQN and PPO baselines
  python experiments/run_all.py --stage validate  # validation-day checks of two design choices
  python experiments/run_all.py --stage bench     # all benchmarks on evaluation seeds 200-229
  python experiments/run_all.py --stage timing    # decision times, single process
  python experiments/run_all.py --stage report    # traces, tables, figures, numbers.json
  python experiments/run_all.py                   # everything, in order

CPU only. Approximate wall time on a 16-core desktop: tune 15 min, train 45 min,
validate 10 min, bench 3-4 h, timing a few minutes, report 1 min.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CORE = None  # read from results/tuning.json after the tune stage


def run(args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY] + args, check=True, cwd=os.path.join(HERE, ".."))


def bench(tag, over=None, noise=None, extra=None, seeds="200-229", methods="all"):
    a = [os.path.join(HERE, "run_benchmark.py"), "--tag", tag, "--seeds", seeds,
         "--methods", methods]
    if over:
        a += ["--over", json.dumps(over)]
    if noise is not None:
        a += ["--noise", str(noise)]
    if extra:
        a += extra
    run(a)


SWEEP_METHODS = ("SDQ-Match,SDQ-core,Whittle,Whittle-F,PSO,PSO-F,GA,Lag-SIPP,JSQ,Erlang-C,"
                 "Greedy,Random,DQN,PPO")


FC_METHODS = "SDQ-Match,Whittle-F,PSO-F,Lag-SIPP,PPO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all")
    st = ap.parse_args().stage
    if st in ("all", "tune"):
        run([os.path.join(HERE, "tune.py")])
        run([os.path.join(HERE, "check_convexity.py")])
    if st in ("all", "train"):
        tun = json.load(open(os.path.join(HERE, "..", "results", "tuning.json")))
        core = json.dumps(tun["main"]["SDQ-core"]["best"])
        run([os.path.join(HERE, "train_es.py"), "--place", "load", "--iters", "150",
             "--core_kw", core, "--out", "results/models/sdq_main.npz"])
        # second member of the ensemble (different training seed)
        run([os.path.join(HERE, "train_es.py"), "--place", "load", "--iters", "150", "--seed", "1",
             "--core_kw", core, "--out", "results/models/sdq_main_s1.npz"])
        run([os.path.join(HERE, "train_dqn.py"), "--out", "results/models/dqn_main.npz"])
        run([os.path.join(HERE, "train_rl.py"), "--kind", "e2e", "--iters", "300",
             "--eval_every", "20", "--out", "results/models/ppo_main.npz"])
    if st in ("all", "validate"):
        # model selection of the learned correction on validation days of all five-spot
        # scenarios: core alone, single members, a QASS-objective variant and ensembles
        tun = json.load(open(os.path.join(HERE, "..", "results", "tuning.json")))
        run([os.path.join(HERE, "train_es.py"), "--place", "load", "--iters", "150",
             "--objective", "qass", "--core_kw", json.dumps(tun["main"]["SDQ-core"]["best"]),
             "--out", "results/models/candidates/sdq_qass.npz"])
        run([os.path.join(HERE, "select_correction.py"), "core=", "seed0=sdq_main.npz",
             "seed1=sdq_main_s1.npz", "qass=candidates/sdq_qass.npz",
             "ens01=sdq_main.npz+sdq_main_s1.npz",
             "ensall=sdq_main.npz+sdq_main_s1.npz+candidates/sdq_qass.npz"])
        # validation-day checks behind two design choices (Protocol section)
        run([os.path.join(HERE, "check_inbound.py")])
        run([os.path.join(HERE, "check_switch_mode.py")])
    if st in ("all", "bench", "bench_a"):
        bench("main")
        for lam in (0.12, 0.14, 0.18, 0.20):
            bench(f"load_{lam}", {"lam0": lam}, methods=SWEEP_METHODS)
        for tau in (2, 8, 10):
            bench(f"tau_{tau}", {"tau_rep": tau}, methods=SWEEP_METHODS)
        for p in (20, 30, 60):
            bench(f"pat_{p}", {"mean_patience": float(p)}, methods=SWEEP_METHODS)
        for e in (0.1, 0.2, 0.3, 0.5):
            bench(f"noise_{e}", noise=e, methods=SWEEP_METHODS)
        for prof in ("weekend", "festival"):
            bench(f"profile_{prof}", {"profile": prof}, methods=SWEEP_METHODS)
            bench(f"mismatch_{prof}", {"profile": prof}, methods=SWEEP_METHODS,
                  extra=["--hist_profile", "weekday"])
            bench(f"nodet_mismatch_{prof}", {"profile": prof}, methods=FC_METHODS,
                  extra=["--hist_profile", "weekday", "--no_detector"])
        bench("dist_negbin", {"arrival": "negbin"}, methods=SWEEP_METHODS)
        bench("dist_lognormal", {"service_dist": "lognormal"}, methods=SWEEP_METHODS)
        bench("dist_uniformpat", {"patience_dist": "uniform"}, methods=SWEEP_METHODS)
    if st in ("all", "bench", "bench_b"):
        for S in (10, 20, 40):
            bench(f"scale_{S}", {"S": S, "N": 2 * S}, seeds="200-209",
                  methods="SDQ-Match,SDQ-core,Whittle,Whittle-F,PSO,PSO-F,GA,Lag-SIPP,JSQ,"
                          "Erlang-C,Greedy,Random")
    if st in ("all", "timing"):
        # decision times of all methods, measured sequentially under equal conditions
        run([os.path.join(HERE, "time_decisions.py")])
    if st in ("all", "report"):
        run([os.path.join(HERE, "make_figures.py")])
        run([os.path.join(HERE, "draw_study_area.py")])
        run([os.path.join(HERE, "gen_report.py")])
        run([os.path.join(HERE, "record_traces.py")])
        run([os.path.join(HERE, "make_extra_figures.py")])


if __name__ == "__main__":
    main()
