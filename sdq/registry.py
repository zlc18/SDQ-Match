"""Method registry: name -> factory(cfg, models) for the benchmark."""
from __future__ import annotations

import json
import os

import numpy as np

from . import policies as P
from .dqn import DQNPolicy
from .forecast import Forecaster
from .rl import SDQMatch, PPOE2E

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "models")


def load_actor(path):
    z = np.load(path, allow_pickle=False)
    n = len([k for k in z.files if k.startswith("arr_")])
    meta = {k: z[k].item() if z[k].shape == () else z[k] for k in z.files if not k.startswith("arr_")}
    return [z[f"arr_{i}"] for i in range(n)], meta


def _model(name, scenario):
    for cand in (f"{name}_{scenario}.npz", f"{name}_main.npz"):
        p = os.path.join(MODEL_DIR, cand)
        if os.path.exists(p):
            return load_actor(p)
    return None, None


TUNING = os.path.join(os.path.dirname(__file__), "..", "results", "tuning.json")
SDQ_PLACE = "load"


def tuned(name, scenario="main"):
    """Hyperparameters selected on the validation seeds by experiments/tune.py."""
    if not os.path.exists(TUNING):
        return {}
    t = json.load(open(TUNING))
    return dict(t.get(scenario, t.get("main", {})).get(name, {}).get("best", {}))


def build(name: str, cfg, scenario: str = "main", **kw):
    fc = Forecaster(cfg, noise=kw.get("noise", 0.0), hist_cfg=kw.get("hist_cfg"),
                    adaptive=kw.get("adaptive", True))
    tp = tuned(name)
    SDQ_CORE_KW = tuned("SDQ-core")
    if name == "Random":
        return P.RandomPolicy(cfg)
    if name == "Greedy":
        return P.GreedyPolicy(cfg)
    if name == "JSQ":
        return P.JSQPolicy(cfg)
    if name == "Erlang-C":
        return P.StaticErlangC(cfg, **tp)
    if name == "Whittle":
        return P.WhittlePolicy(cfg, fc, **{"rate": "observed", **tp})
    if name == "Whittle-F":
        return P.WhittlePolicy(cfg, fc, **{"rate": "forecast", **tp})
    if name == "Lag-SIPP":
        return P.LagSIPP(cfg, fc)
    if name == "PSO":
        return P.PSOPolicy(cfg, fc, **{"rate": "observed", **tp})
    if name == "GA":
        return P.GAPolicy(cfg, fc, **{"rate": "observed", **tp})
    if name == "PSO-F":
        return P.PSOPolicy(cfg, fc, **{"rate": "forecast", **tp})
    if name == "GA-F":
        return P.GAPolicy(cfg, fc, rate="forecast", horizon=tuned("PSO-F").get("horizon", 20))
    if name == "DQN":
        actor, _ = _model("dqn", scenario)
        pol = DQNPolicy(cfg, fc, actor)
        return pol
    if name == "PPO":
        actor, _ = _model("ppo", scenario)
        return PPOE2E(cfg, fc, actor)
    if name == "SDQ-Match":        # queueing core + learned per-spot load correction (ES)
        actor, meta = _model("sdq", scenario)
        # ensemble of independently trained corrections (selected on validation days)
        extra = os.path.join(MODEL_DIR, "sdq_main_s1.npz")
        if actor is not None and os.path.exists(extra):
            actor = (actor, load_actor(extra)[0])
        place = str(meta["place"]) if meta and "place" in meta else SDQ_PLACE
        core_kw = json.loads(str(meta["core_kw"])) if meta and "core_kw" in meta else SDQ_CORE_KW
        from .scenarios import get_scenario
        train_scen = str(meta["scenario"]) if meta and "scenario" in meta else "main"
        pol = SDQMatch(cfg, fc, actor, core_kw=core_kw, place=place,
                       train_tau=get_scenario(train_scen).tau_rep)
        pol.name = "SDQ-Match"
        return pol
    if name == "SDQ-core":         # ablation: the queueing core without the learned correction
        pol = P.SDQTransient(cfg, fc, **SDQ_CORE_KW)
        pol.name = "SDQ-core"
        return pol
    # ---- further ablations, each applied to the queueing core
    if name == "abl-no-lead":
        return P.SDQTransient(cfg, fc, **{**SDQ_CORE_KW, "lead": False})
    if name == "abl-steady":       # steady-state Erlang-A instead of the transient cost
        return P.SDQCore(cfg, fc, coverage=False, h=tuned("Whittle-F").get("h", 20.0))
    if name == "abl-surrogate":    # pressure-per-server surrogate pi/(c+1)
        return P.SDQCore(cfg, fc, coverage=False, exact=False, h=5.0)
    if name == "abl-no-forecast":  # present observed rate in both phases
        class _Obs(Forecaster):
            def rate_at(self, ts):
                return np.repeat(self.env.observe()["recent_rate"][None], len(ts), 0)
        return P.SDQTransient(cfg, _Obs(cfg), **{**SDQ_CORE_KW, "lead": False})
    raise ValueError(name)


# tier 1: published methods as specified in their sources (Whittle, PSO, GA: observed rates)
# tier 2: forecast-augmented methods sharing SDQ-Match's forecaster
BASELINES_T1 = ["Random", "Greedy", "JSQ", "Erlang-C", "Whittle", "PSO", "GA", "DQN"]
BASELINES_T2 = ["Lag-SIPP", "Whittle-F", "PSO-F", "PPO"]
BASELINES = BASELINES_T1 + BASELINES_T2
OURS = "SDQ-Match"
