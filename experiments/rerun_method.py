"""Re-run one method in every benchmark scenario and merge its rows into the existing
per_seed.csv files (all other methods are deterministic and unchanged).

Usage: python experiments/rerun_method.py SDQ-Match
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(tag, method, over=None, noise=None, extra=None, seeds="200-229"):
    a = [PY, os.path.join(HERE, "run_benchmark.py"), "--tag", tag, "--seeds", seeds,
         "--methods", method, "--merge", "--workers", "14"]
    if over:
        a += ["--over", json.dumps(over)]
    if noise is not None:
        a += ["--noise", str(noise)]
    if extra:
        a += extra
    print(">>", tag, flush=True)
    subprocess.run(a, check=True, cwd=os.path.join(HERE, ".."))


def main():
    m = sys.argv[1]
    run("main", m)
    for lam in (0.12, 0.14, 0.18, 0.20):
        run(f"load_{lam}", m, {"lam0": lam})
    for tau in (2, 8, 10):
        run(f"tau_{tau}", m, {"tau_rep": tau})
    for p in (20, 30, 60):
        run(f"pat_{p}", m, {"mean_patience": float(p)})
    for e in (0.1, 0.2, 0.3, 0.5):
        run(f"noise_{e}", m, noise=e)
    for prof in ("weekend", "festival"):
        run(f"profile_{prof}", m, {"profile": prof})
        run(f"mismatch_{prof}", m, {"profile": prof}, extra=["--hist_profile", "weekday"])
        run(f"nodet_mismatch_{prof}", m, {"profile": prof},
            extra=["--hist_profile", "weekday", "--no_detector"])
    run("dist_negbin", m, {"arrival": "negbin"})
    run("dist_lognormal", m, {"service_dist": "lognormal"})
    run("dist_uniformpat", m, {"patience_dist": "uniform"})
    for S in (10, 20, 40):
        run(f"scale_{S}", m, {"S": S, "N": 2 * S}, seeds="200-209")


if __name__ == "__main__":
    main()
