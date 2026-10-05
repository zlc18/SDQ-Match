# SDQ-Match

Source code, trained models and per-day results for the paper

> Li Zhang. *Matching a shared eVTOL fleet to tourist demand in scenic areas with transient
> Erlang-A water-filling.*

SDQ-Match allocates a shared fleet of sightseeing aircraft across the boarding spots of a scenic
area. Each spot is modelled as an Erlang-A (M/M/c+M) queue. For every candidate change in the
supply of a spot, the matcher computes the waiting time the spot will accrue over the next
matching epoch, from the observed number of tourists and in two phases split at the reposition
latency, and allocates the fleet by marginal-cost water-filling. A small per-spot network
corrects the queueing loads.

The repository contains the simulator, SDQ-Match, the twelve baselines of the paper, the
training, tuning and evaluation scripts, and every per-day result behind the tables and figures.
Everything runs on a CPU.

## Installation

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip install -r requirements.txt
```

Tested with Python 3.13, NumPy 2.4, SciPy 1.17, Matplotlib 3.10 and PyTorch 2.11.
PyTorch is needed only to train the DQN and PPO baselines.

## Quick start

Run SDQ-Match and two baselines on three evaluation days of the calibration scenario and print
the paired comparison:

```bash
python experiments/run_benchmark.py --tag demo --seeds 200-202 --methods SDQ-Match,Whittle-F,Whittle
python experiments/eval_metrics.py --tag demo
```

## Reproducing the paper

`results/` already holds every output of the paper, so the tables and figures can be rebuilt
directly:

```bash
python experiments/run_all.py --stage report
```

To rerun the whole study from scratch:

```bash
python experiments/run_all.py
```

or stage by stage:

| Stage | What it does | Time (16-core CPU) |
|---|---|---|
| `tune` | grid search of every tunable method on validation days 100–119 (`results/tuning.json`); convexity check (`results/convexity.json`) | ~15 min |
| `train` | learned correction of SDQ-Match (two training seeds), DQN and PPO baselines (`results/models/`) | ~45 min |
| `validate` | selection of the learned correction and two design checks on validation days | ~10 min |
| `bench` | all 25 scenarios on evaluation days 200–229 (`results/<scenario>/per_seed.csv`) | 3–4 h |
| `timing` | decision time of every method, single process (`results/timing.json`) | a few minutes |
| `report` | `results/numbers.json`, `results/REPORT.md`, LaTeX tables and figures | ~1 min |

Run a stage with `python experiments/run_all.py --stage <name>`.

### Where each result comes from

| Paper | File | Script |
|---|---|---|
| Table 3 (main results) | `results/tables/main.tex` | `gen_report.py` |
| Table 4 (paired statistics) | `results/tables/significance.tex` | `gen_report.py` |
| Table 5 (all scenarios) | `results/tables/scenarios.tex` | `gen_report.py` |
| Table 6 (learned correction) | `results/tables/es_variant.tex` | `gen_report.py` |
| Fig 4 (per-day gain) | `results/figures/fig_daily_gain.png` | `make_extra_figures.py` |
| Fig 5 (time-of-day dynamics) | `results/figures/fig_dynamics.png` | `record_traces.py`, `make_extra_figures.py` |
| Fig 6 (ablation) | `results/figures/fig_ablation.png` | `make_figures.py` |
| Fig 7 (training curves) | `results/figures/fig_learning.png` | `make_figures.py` |
| Fig 8 (sensitivity) | `results/figures/fig_sweeps.png` | `make_figures.py` |
| Fig 9 (gains across scenarios) | `results/figures/fig_forest.png` | `make_extra_figures.py` |
| Convexity check (Proposition 1) | `results/convexity.json` | `check_convexity.py` |

## Layout

```
sdq/
  env.py          discrete-event simulator, common random numbers per seed
  demand.py       daily demand profiles with a moving spatial hotspot
  forecast.py     historical-profile forecaster with online level correction and pattern test
  queueing.py     stationary Erlang-A / Erlang-C laws, marginal water-filling
  transient.py    two-phase transient Erlang-A cost (matrix exponential)
  policies.py     baselines (Random, Greedy, JSQ, Erlang-C, Whittle, Lag-SIPP, PSO, GA)
                  and the queueing core of SDQ-Match
  rl.py           learned load correction of SDQ-Match, end-to-end PPO baseline
  dqn.py          DQN baseline
  stats.py        Wilcoxon signed-rank test, Holm correction, paired bootstrap CI
  registry.py     method name -> configured policy (reads results/tuning.json)
  scenarios.py    named scenarios
experiments/      tuning, training, benchmark, statistics, tables and figures
results/          per-day results, selected hyperparameters, trained models, tables, figures
```

## Evaluation protocol

* Seeds. Historical days for the forecaster: 9000–9029. Training days: 10000 and above.
  Validation days (all hyperparameter and model choices): 100–119. Evaluation days: 200–229
  (200–209 for the 10-, 20- and 40-spot networks). No evaluation day is used for any choice.
* Common random numbers. A seed fixes the whole simulated day (arrivals, patience, flight
  durations, demand multiplier and peak timing), so all methods face the same days and every
  comparison is paired.
* Statistics. Two-sided Wilcoxon signed-rank test on the per-day QASS with Holm correction over
  the baselines of a scenario, paired bootstrap 95% confidence interval of the relative gain,
  paired effect size d_z, and win/tie/loss counts.
* Baselines. Tier 1: published methods as specified (Random, Greedy, JSQ, static Erlang-C,
  Whittle index, PSO, GA, DQN). Tier 2: the same families given the forecaster of SDQ-Match
  (Lag-SIPP, Whittle-F, PSO-F, end-to-end PPO). Every tunable method receives a grid search on
  the same validation days (`experiments/tune.py`).

## License

MIT, see `LICENSE`.
