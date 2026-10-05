"""Named experimental scenarios. ``main`` is the calibration scenario of the paper."""
from __future__ import annotations

from .env import EnvConfig

MAIN = dict(S=5, N=10, T=720, lam0=0.16, base=0.6, amp=3.0, mean_service=7.0,
            mean_patience=45.0, tau_rep=5, kappa=5, profile="weekday")


def get_scenario(name: str, **over) -> EnvConfig:
    d = dict(MAIN)
    if name == "main":
        pass
    elif name.startswith("S"):          # e.g. S20 -> 20 spots, 40 aircraft
        S = int(name[1:])
        d.update(S=S, N=2 * S)
    elif name in ("weekday", "weekend", "festival"):
        d.update(profile=name)
    else:
        raise ValueError(name)
    d.update(over)
    return EnvConfig(**d)
