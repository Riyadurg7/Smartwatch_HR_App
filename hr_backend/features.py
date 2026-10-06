"""
Feature engineering for smartwatch heart-rate data.

Objective 2 (support) + Objective 4 inputs: rolling averages, daily/hourly
statistics, circadian profile, resting/max HR, a BPM-based variability proxy,
and peak detection for exercise/recovery curves.

Works on the cleaned frame produced by preprocessing.preprocess(), which has
columns: timestamp, hr_resampled, hr_smooth, is_filled.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass
class HRStats:
    mean_hr: float = np.nan
    median_hr: float = np.nan
    std_hr: float = np.nan
    min_hr: float = np.nan
    max_hr: float = np.nan          # robust max (99th pct)
    abs_max_hr: float = np.nan      # true observed max
    resting_hr: float = np.nan      # headline awake resting HR (daytime low-activity estimate)
    overnight_low_hr: float = np.nan  # lowest sustained overnight HR (sleep) - normal to be low
    resting_hr_1h: float = np.nan   # lowest sustained 1-hour mean (whole day, incl. sleep)
    hrv_proxy_ms: float = np.nan    # RMSSD-style proxy from BPM differences (NOT clinical HRV)
    p05: float = np.nan
    p95: float = np.nan
    resting_method: str = "daytime 10th percentile (no activity data)"

    def to_dict(self) -> dict:
        return asdict(self)


def _series(clean: pd.DataFrame, col: str = "hr_smooth") -> pd.Series:
    s = clean.set_index("timestamp")[col].dropna()
    return s


def compute_stats(clean: pd.DataFrame) -> HRStats:
    """Scalar summary statistics over the whole recording."""
    s = _series(clean, "hr_resampled")
    if s.empty:
        return HRStats()

    stats = HRStats(
        mean_hr=round(float(s.mean()), 1),
        median_hr=round(float(s.median()), 1),
        std_hr=round(float(s.std()), 1),
        min_hr=round(float(s.min()), 1),
        abs_max_hr=round(float(s.max()), 1),
        max_hr=round(float(s.quantile(0.99)), 1),
        p05=round(float(s.quantile(0.05)), 1),
        p95=round(float(s.quantile(0.95)), 1),
    )

    smooth = _series(clean, "hr_smooth")

    # Headline RESTING HR ~ like a wearable's "resting heart rate": measured
    # during inactive but AWAKE periods. We approximate this as the 10th
    # percentile of daytime (08:00-22:00) readings, which excludes both exertion
    # (high) and deep sleep (very low). Falls back to overall 10th pct if no
    # daytime data is present.
    daytime = smooth[(smooth.index.hour >= 8) & (smooth.index.hour < 22)]
    base = daytime if len(daytime) >= 50 else smooth
    stats.resting_hr = round(float(base.quantile(0.10)), 1)

    # Overnight low: sustained low during the usual sleep window. It is NORMAL
    # for this to sit well below 60 bpm; reported separately, never a diagnosis.
    night = smooth[(smooth.index.hour >= 0) & (smooth.index.hour < 6)]
    if len(night) >= 20:
        stats.overnight_low_hr = round(float(night.rolling(60, min_periods=10).mean().min()), 1)
        if np.isnan(stats.overnight_low_hr):
            stats.overnight_low_hr = round(float(night.quantile(0.05)), 1)

    # Lowest sustained 1-hour rolling mean across the whole day (incl. sleep).
    try:
        hourly_roll = smooth.rolling("1h", min_periods=1).mean()
        stats.resting_hr_1h = round(float(hourly_roll.min()), 1)
    except Exception:
        stats.resting_hr_1h = stats.resting_hr

    # Variability proxy (RMSSD of successive BPM diffs, expressed in ms-equivalent).
    stats.hrv_proxy_ms = round(_hrv_proxy(s), 1)
    return stats


def _hrv_proxy(bpm: pd.Series) -> float:
    """
    A variability proxy derived from beat-per-minute differences.

    True HRV (RMSSD/SDNN) needs inter-beat (RR) intervals, which exported BPM
    data does not contain. We convert each BPM sample to an approximate mean RR
    interval (60000 / bpm ms) and compute the RMSSD of successive differences.
    This is an APPROXIMATION for trend-tracking only, not a clinical HRV value.
    """
    bpm = bpm.dropna()
    if len(bpm) < 3:
        return float("nan")
    rr_ms = 60000.0 / bpm.clip(lower=1)
    diffs = np.diff(rr_ms.values)
    return float(np.sqrt(np.mean(diffs ** 2)))


def rolling_features(clean: pd.DataFrame, windows=("5min", "30min", "1h")) -> pd.DataFrame:
    """Add moving-average columns (Objective 2: moving averages)."""
    out = clean.copy().set_index("timestamp")
    base = out["hr_smooth"]
    for w in windows:
        out[f"ma_{w}"] = base.rolling(w, min_periods=1).mean()
    return out.reset_index()


def daily_stats(clean: pd.DataFrame) -> pd.DataFrame:
    """Per-day summary table (Objective 2: daily statistics / summary dashboards)."""
    s = _series(clean, "hr_resampled")
    if s.empty:
        return pd.DataFrame()
    g = s.groupby(s.index.date)
    daily = pd.DataFrame(
        {
            "date": list(g.groups.keys()),
            "mean_hr": g.mean().round(1).values,
            "min_hr": g.min().round(1).values,
            "max_hr": g.max().round(1).values,
            "low_hr": g.quantile(0.05).round(1).values,      # includes sleep
            "resting_hr": _daytime_low(s),
            "std_hr": g.std().round(1).values,
            "n_samples": g.size().values,
        }
    )
    return daily


def _daytime_low(s: pd.Series) -> np.ndarray:
    """Fallback per-day resting HR without activity data: 10th pct of 08:00-22:00."""
    day = s[(s.index.hour >= 8) & (s.index.hour < 22)]
    q = day.groupby(day.index.date).quantile(0.10).round(1)
    return np.array([q.get(d, np.nan) for d in sorted(set(s.index.date))])


def circadian_profile(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Average heart rate by hour-of-day across the whole recording
    (Objective 2: circadian trends).
    """
    s = _series(clean, "hr_resampled")
    if s.empty:
        return pd.DataFrame()
    by_hour = s.groupby(s.index.hour)
    prof = pd.DataFrame(
        {
            "hour": list(range(24)),
        }
    ).merge(
        pd.DataFrame(
            {
                "hour": by_hour.mean().index,
                "mean_hr": by_hour.mean().round(1).values,
                "std_hr": by_hour.std().round(1).values,
                "n": by_hour.size().values,
            }
        ),
        on="hour",
        how="left",
    )
    return prof


def detect_exertion_peaks(
    clean: pd.DataFrame,
    resting_hr: Optional[float] = None,
    elevation_bpm: float = 30.0,
    min_duration_min: int = 5,
    merge_gap_min: int = 5,
) -> pd.DataFrame:
    """
    Detect sustained exertion EPISODES (candidate exercise bouts), not every
    local wiggle (Objective 2: exercise recovery curves).

    An episode is a stretch where smoothed HR stays above
    `resting_hr + elevation_bpm` for at least `min_duration_min`. Episodes
    separated by less than `merge_gap_min` are merged. For each episode we
    report start/end/duration, peak HR, and HR recovery (drop over 1 and 2
    minutes after the peak) - a higher recovery suggests better fitness.
    """
    s = _series(clean, "hr_smooth")
    if len(s) < 10:
        return pd.DataFrame()

    if resting_hr is None or np.isnan(resting_hr):
        resting_hr = float(s.quantile(0.10))
    threshold = resting_hr + elevation_bpm

    above = s > threshold
    if not above.any():
        return pd.DataFrame()

    # Group contiguous "above" runs, merging short gaps.
    step_s = s.index.to_series().diff().dt.total_seconds().median() or 60.0
    merge_gap = pd.Timedelta(minutes=merge_gap_min)
    min_dur = pd.Timedelta(minutes=min_duration_min)

    times = s.index[above]
    if len(times) == 0:
        return pd.DataFrame()

    episodes = []
    run_start = prev = times[0]
    for t in times[1:]:
        if t - prev > merge_gap:
            episodes.append((run_start, prev))
            run_start = t
        prev = t
    episodes.append((run_start, prev))

    rows = []
    for start, end in episodes:
        if end - start < min_dur:
            continue
        seg = s.loc[start:end]
        peak_t = seg.idxmax()
        rows.append(
            {
                "start": start,
                "end": end,
                "duration_min": round((end - start).total_seconds() / 60, 1),
                "peak_hr": round(float(seg.max()), 1),
                "mean_hr": round(float(seg.mean()), 1),
                "recovery_1min_bpm": _recovery(s, peak_t, minutes=1),
                "recovery_2min_bpm": _recovery(s, peak_t, minutes=2),
            }
        )
    return pd.DataFrame(rows)


def _recovery(s: pd.Series, peak_time, minutes: int) -> float:
    """HR drop (bpm) from peak to `minutes` after the peak. Higher = fitter."""
    try:
        after = s.loc[peak_time: peak_time + pd.Timedelta(minutes=minutes)]
        if len(after) < 2:
            return float("nan")
        return round(float(after.iloc[0] - after.iloc[-1]), 1)
    except Exception:
        return float("nan")


def windowed_feature_matrix(clean: pd.DataFrame, window: str = "10min") -> pd.DataFrame:
    """
    Build a per-window feature matrix used by the ML module (anomaly detection
    and clustering). Each row summarizes one time window.

    Columns: window_start, mean, std, min, max, range, slope, hour, n
    """
    s = _series(clean, "hr_resampled")
    if s.empty:
        return pd.DataFrame()

    rows = []
    for win_start, chunk in s.resample(window):
        c = chunk.dropna()
        if len(c) == 0:
            continue
        if len(c) >= 2:
            x = np.arange(len(c))
            slope = float(np.polyfit(x, c.values, 1)[0])
        else:
            slope = 0.0
        rows.append(
            {
                "window_start": win_start,
                "mean": float(c.mean()),
                "std": float(c.std(ddof=0)),
                "min": float(c.min()),
                "max": float(c.max()),
                "range": float(c.max() - c.min()),
                "slope": slope,
                # largest step between consecutive samples: exercise ramps HR up
                # gradually, sensor glitches / abrupt events jump
                "max_jump": float(c.diff().abs().max()) if len(c) >= 2 else 0.0,
                "hour": int(win_start.hour),
                "n": int(len(c)),
            }
        )
    return pd.DataFrame(rows)
