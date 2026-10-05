"""Daily demand profiles with a moving spatial hotspot.

The arrival rate at spot s and step t is

    lambda_s(t) = lam0 * w_s * m_s(t) * D,

where w_s is a fixed popularity weight, m_s(t) is the profile multiplier of
the group the spot belongs to, and D is a day-level demand multiplier drawn
once per simulated day. Peak times are jittered per day, so a forecast built
from historical days is informative but never exact.
"""
from __future__ import annotations

import numpy as np

PROFILES = ("weekday", "weekend", "festival")


def popularity_weights(S: int) -> np.ndarray:
    """Fixed, scenario-level popularity weights (mean 1), identical for all seeds."""
    rng = np.random.default_rng(12345 + S)
    w = rng.uniform(0.7, 1.3, size=S)
    return w / w.mean()


def _bump(t: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def spot_groups(S: int, profile: str) -> list[tuple[np.ndarray, float]]:
    """Return a list of (spot_indices, peak_fraction_of_day) for the profile."""
    idx = np.arange(S)
    n_a = max(1, int(round(0.4 * S)))
    A, B = idx[:n_a], idx[n_a:]
    if profile == "weekday":
        # entrance/museum in the morning, ridge-line and valley in the afternoon
        return [(A, 0.30), (B, 0.70)]
    if profile == "weekend":
        # families move the other way: valley spots first
        return [(B, 0.30), (A, 0.70)]
    if profile == "festival":
        # a central pair at midday disperses to the flanking spots in the evening
        mid = S // 2
        centre = idx[max(0, mid - 1): mid + 1] if S >= 3 else idx[:1]
        flank = np.setdiff1d(idx, centre)
        return [(centre, 0.50), (flank, 0.80)]
    raise ValueError(f"unknown profile {profile}")


def rate_matrix(S: int, T: int, lam0: float, profile: str,
                base: float = 0.35, amp: float = 2.0, width_frac: float = 0.08,
                peak_shift: np.ndarray | None = None, day_mult: float = 1.0) -> np.ndarray:
    """Return the (T, S) matrix of arrival rates per step."""
    t = np.arange(T, dtype=float)
    w = popularity_weights(S)
    m = np.full((T, S), base)
    for g, (spots, frac) in enumerate(spot_groups(S, profile)):
        shift = 0.0 if peak_shift is None else float(peak_shift[g])
        bump = _bump(t, frac * T + shift, width_frac * T)
        m[:, spots] += amp * bump[:, None]
    return lam0 * day_mult * w[None, :] * m


def day_parameters(rng: np.random.Generator, day_sigma: float, peak_jitter: float):
    """Day-level demand multiplier and per-group peak shifts."""
    day_mult = float(np.exp(rng.normal(-0.5 * day_sigma ** 2, day_sigma)))
    peak_shift = rng.normal(0.0, peak_jitter, size=2)
    return day_mult, peak_shift
