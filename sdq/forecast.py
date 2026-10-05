"""Demand forecaster available to every forecast-using policy.

The forecast is the historical arrival profile, estimated from a set of past
simulated days (disjoint seeds), smoothed in time, and rescaled online by the
ratio of today's cumulative arrivals to the historical cumulative arrivals.
No policy ever sees the true rate of the day it is evaluated on.
"""
from __future__ import annotations

import numpy as np

from .env import EnvConfig, realise_day

HIST_SEEDS = range(9000, 9030)   # 30 historical days


def _smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    out = np.empty_like(x, dtype=float)
    for s in range(x.shape[1]):
        out[:, s] = np.convolve(pad[:, s], k, mode="valid")
    return out


_CACHE: dict = {}


def historical_profile(cfg: EnvConfig, sigma: float = 8.0) -> np.ndarray:
    key = (cfg.S, cfg.T, cfg.lam0, cfg.profile, cfg.amp, cfg.base, cfg.width_frac,
           cfg.day_sigma, cfg.peak_jitter, cfg.arrival, cfg.nb_size, sigma)
    if key not in _CACHE:
        acc = np.zeros((cfg.T, cfg.S))
        for sd in HIST_SEEDS:
            day = realise_day(cfg, sd)
            for s, (t_arr, _, _) in enumerate(day.arrivals):
                acc[:, s] += np.bincount(t_arr, minlength=cfg.T)[: cfg.T]
        _CACHE[key] = _smooth(acc / len(HIST_SEEDS), sigma)
    return _CACHE[key]


class Forecaster:
    """Lead-time rate forecast lambda_hat_s(t+k) from history plus online level correction."""

    def __init__(self, cfg: EnvConfig, hist_cfg: EnvConfig | None = None,
                 noise: float = 0.0, online: bool = True, shrink: float = 30.0,
                 adaptive: bool = True, alpha: float = 1e-9, det_window: int = 60,
                 spot_window: int = 20, spot_prior: float = 3.0,
                 switch_mode: str = "persist", persist_window: int = 30):
        self.cfg = cfg
        self.hist = historical_profile(hist_cfg or cfg)
        self.cum_hist = np.concatenate([[0.0], np.cumsum(self.hist.sum(1))])
        self.noise = noise
        self.online = online
        self.shrink = shrink
        self.adaptive = adaptive
        self.alpha, self.det_window = alpha, det_window
        self.spot_window, self.spot_prior = spot_window, spot_prior
        # after the pattern test fires: "ratio" rescales the historical profile per spot,
        # "persist" forecasts every horizon with the recently observed per-spot rate
        self.switch_mode, self.persist_window = switch_mode, persist_window
        self.pad = 200

    def reset(self, env, seed: int = 0):
        from scipy.stats import chi2
        self.env = env
        T, S = self.cfg.T, self.cfg.S
        rng = np.random.default_rng(10_000_019 + seed)
        if self.noise > 0:
            # multiplicative error, smooth in time (an error in the profile, not white noise)
            z = _smooth(rng.normal(size=(T + self.pad, S)), 10.0)
            z /= max(1e-9, z.std())
            self.err = np.clip(1.0 + self.noise * z, 0.05, None)
        else:
            self.err = np.ones((T + self.pad, S))
        self.switched = False
        self.switch_time = None
        self._crit = chi2.ppf(1 - self.alpha, S)
        self._cache_t, self._cache_r = -1, None

    def level(self) -> float:
        if not self.online:
            return 1.0
        t = self.env.t
        obs = self.env.arr_hist[:t].sum()
        exp = self.cum_hist[t]
        k = self.shrink
        return float(np.clip((obs + k) / (exp + k), 0.5, 2.0))

    def spot_scale(self) -> np.ndarray:
        """Per-spot multiplier of the historical profile.

        By default every spot gets the global level. A Pearson chi-square test
        compares the last ``det_window`` steps of per-spot arrivals with the
        level-corrected history; once it rejects at level ``alpha`` (the day's
        spatial pattern differs from history), the forecaster switches for the
        rest of the day to per-spot ratios over the last ``spot_window`` steps.
        """
        t = self.env.t
        if t == self._cache_t:
            return self._cache_r
        L = self.level()
        r = np.full(self.cfg.S, L)
        if self.adaptive:
            if not self.switched and t >= 20:
                lo = max(0, t - self.det_window)
                obs = self.env.arr_hist[lo:t].sum(0)
                exp = self.hist[lo:t].sum(0) * L
                if ((obs - exp) ** 2 / (exp + 1.0)).sum() > self._crit:
                    self.switched, self.switch_time = True, t
            if self.switched:
                lo = max(0, t - self.spot_window)
                obs = self.env.arr_hist[lo:t].sum(0)
                exp = self.hist[lo:t].sum(0)
                k = self.spot_prior
                r = np.clip((obs + k) / (exp + k), 0.25, 4.0)
        self._cache_t, self._cache_r = t, r
        return r

    def rate_at(self, ts: np.ndarray) -> np.ndarray:
        """Forecast rates at absolute steps ts, shape (len(ts), S)."""
        ts = np.clip(ts, 0, self.cfg.T - 1)
        scale = self.spot_scale()
        if self.switched and self.switch_mode == "persist":
            t = self.env.t
            lo = max(0, t - self.persist_window)
            rate = self.env.arr_hist[lo:t].sum(0) / max(1, t - lo)
            return np.repeat(rate[None, :], len(ts), 0) * self.err[ts]
        return self.hist[ts] * scale[None, :] * self.err[ts]

    def mean_rate(self, t0: int, t1: int) -> np.ndarray:
        ts = np.arange(t0, max(t0 + 1, t1))
        return self.rate_at(ts).mean(0)
