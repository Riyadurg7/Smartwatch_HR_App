"""
Sleep- and activity-aware analysis.

Heart rate means little without context: 120 bpm while running is normal,
120 bpm while sitting still is worth noticing. This module joins per-minute
heart rate with activity intensity, steps and sleep (observed from Fitbit
companion files, or - for intensity - estimated by the trained activity model
when only heart rate is available) and derives context-aware statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .fitbit import INTENSITY_LABELS

ACTIVITY_NAMES = {0: "sedentary", 1: "light", 2: "active"}  # model classes


@dataclass
class ContextResult:
    available: bool = False
    intensity_source: str = "none"      # "observed" | "estimated" | "none"
    sleep_source: str = "none"          # "observed" | "assumed (00:00-06:00)"
    resting_hr: float = np.nan          # HR during sustained awake rest
    n_rest_minutes: int = 0
    sleep_hr_mean: float = np.nan
    sleep_hr_low: float = np.nan        # lowest 30-min mean while asleep
    sleep_minutes: int = 0
    hr_by_activity: list = field(default_factory=list)   # rows: activity, minutes, pct, mean_hr, median_hr
    elevated_rest_episodes: list = field(default_factory=list)
    minutes: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("minutes", None)
        return d


def minute_hr_from_clean(canon: pd.DataFrame) -> pd.DataFrame:
    """
    Per-minute HR summary from a standardized frame (timestamp, hr) - works for
    any uploaded CSV, not just Fitbit.
    """
    d = canon.dropna(subset=["timestamp", "hr"]).copy()
    d = d[(d["hr"] >= 25) & (d["hr"] <= 240)]
    d["minute"] = d["timestamp"].dt.floor("min")
    g = d.groupby("minute")["hr"]
    out = g.agg(hr_mean="mean", hr_std="std", hr_min="min", hr_max="max", hr_n="count").reset_index()
    out["hr_std"] = out["hr_std"].fillna(0.0)
    return out


def _filter_user(df: Optional[pd.DataFrame], user_id) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if "Id" in df.columns and user_id is not None:
        df = df[df["Id"].astype(str) == str(user_id)]
    return df.drop(columns=[c for c in ["Id"] if c in df.columns])


def build_context(
    minute_hr: pd.DataFrame,
    intensity: Optional[pd.DataFrame] = None,
    steps: Optional[pd.DataFrame] = None,
    sleep: Optional[pd.DataFrame] = None,
    user_id=None,
    activity_model=None,
) -> ContextResult:
    """
    Join context onto per-minute HR and compute context-aware statistics.

    intensity/steps/sleep: tidy frames from hr_backend.fitbit loaders
    (columns Id?, minute, intensity|steps|sleep_state).
    activity_model: optional ActivityModel used when intensity is missing.
    """
    res = ContextResult()
    if minute_hr is None or minute_hr.empty:
        return res
    m = minute_hr.copy()

    inten = _filter_user(intensity, user_id)
    stp = _filter_user(steps, user_id)
    slp = _filter_user(sleep, user_id)

    # ---- activity level (3 classes: sedentary / light / active) ----
    if inten is not None and len(inten):
        m = m.merge(inten, on="minute", how="left")
        m["activity"] = m["intensity"].map({0: 0, 1: 1, 2: 2, 3: 2})
        res.intensity_source = "observed"
    elif activity_model is not None:
        pred = activity_model.predict(m)
        m["activity"] = pred["activity"].values
        m["p_active"] = pred["p_active"].values
        res.intensity_source = "estimated"
    if stp is not None and len(stp):
        m = m.merge(stp, on="minute", how="left")

    # ---- sleep ----
    if slp is not None and len(slp):
        m = m.merge(slp, on="minute", how="left")
        m["is_asleep"] = m["sleep_state"].isin([1, 2])
        res.sleep_source = "observed"
    else:
        h = m["minute"].dt.hour
        m["is_asleep"] = (h >= 0) & (h < 6)
        res.sleep_source = "assumed (00:00-06:00)"

    res.available = "activity" in m.columns
    m = m.sort_values("minute").reset_index(drop=True)
    res.minutes = m

    # ---- sleep HR ----
    asleep = m.loc[m["is_asleep"], ["minute", "hr_mean"]].set_index("minute")["hr_mean"]
    res.sleep_minutes = int(len(asleep))
    if len(asleep) >= 30:
        res.sleep_hr_mean = round(float(asleep.mean()), 1)
        low = asleep.rolling("30min", min_periods=20).mean().min()
        res.sleep_hr_low = round(float(low), 1) if pd.notna(low) else np.nan

    if not res.available:
        return res

    # ---- resting HR: sustained (10 min) step-free sedentary, awake ----
    s = m.set_index("minute")
    sed = s["activity"] == 0
    if "steps" in s:
        sed &= s["steps"].fillna(0) == 0
    sustained = sed.astype(float).rolling("10min").min() == 1
    rest_mask = sustained & ~s["is_asleep"]
    res.minutes["is_rest"] = rest_mask.values
    rest = s.loc[rest_mask, "hr_mean"]
    res.n_rest_minutes = int(len(rest))
    if len(rest) >= 30:
        res.resting_hr = round(float(rest.quantile(0.25)), 1)

    # ---- HR by activity level ----
    total = len(m)
    rows = []
    for a, name in ACTIVITY_NAMES.items():
        sub = m.loc[(m["activity"] == a) & ~m["is_asleep"], "hr_mean"]
        rows.append(dict(activity=name, minutes=int(len(sub)),
                         pct=round(100 * len(sub) / total, 1) if total else 0.0,
                         mean_hr=round(float(sub.mean()), 1) if len(sub) else np.nan,
                         median_hr=round(float(sub.median()), 1) if len(sub) else np.nan))
    rows.append(dict(activity="asleep", minutes=res.sleep_minutes,
                     pct=round(100 * res.sleep_minutes / total, 1) if total else 0.0,
                     mean_hr=res.sleep_hr_mean, median_hr=round(float(asleep.median()), 1) if len(asleep) else np.nan))
    res.hr_by_activity = rows

    # ---- elevated HR while at rest (context-aware observation) ----
    if not np.isnan(res.resting_hr):
        res.elevated_rest_episodes = _elevated_at_rest(s, sustained & ~s["is_asleep"], res.resting_hr)
    return res


def _elevated_at_rest(s: pd.DataFrame, rest_mask: pd.Series, resting_hr: float,
                      margin: float = 30.0, min_minutes: int = 10) -> list:
    """
    Find stretches of >= min_minutes where the person is still (sustained
    sedentary, awake, no steps) yet HR is > resting_hr + margin (and > 100).
    These are worth a look (stress, caffeine, illness - or a sensor issue).
    """
    thr = max(100.0, resting_hr + margin)
    flag = rest_mask & (s["hr_mean"] > thr)
    if not flag.any():
        return []
    times = s.index[flag]
    episodes, start, prev = [], times[0], times[0]
    for t in times[1:]:
        if t - prev > pd.Timedelta(minutes=2):
            episodes.append((start, prev)); start = t
        prev = t
    episodes.append((start, prev))
    out = []
    for a, b in episodes:
        dur = (b - a).total_seconds() / 60 + 1
        if dur >= min_minutes:
            seg = s.loc[a:b, "hr_mean"]
            out.append(dict(start=str(a), end=str(b), minutes=int(dur),
                            mean_hr=round(float(seg.mean()), 1), max_hr=round(float(seg.max()), 1)))
    return out


def add_window_context(feat: pd.DataFrame, ctx: ContextResult, window: str) -> pd.DataFrame:
    """
    Add context columns to the windowed feature matrix:
      mean_activity, steps_sum, asleep_frac, expected_hr and residual features.

    expected_hr = this person's median HR for the activity level of each minute
    (sleep treated as its own level). residual = actual - expected, so a window
    of hard exercise with appropriately high HR has a NORMAL residual, while
    high HR at rest has a large one.
    """
    if feat.empty or not ctx.available or ctx.minutes.empty:
        return feat
    m = ctx.minutes.copy()
    level = m["activity"].astype("float").where(~m["is_asleep"], -1)
    med = m.groupby(level)["hr_mean"].median()
    m["expected_hr"] = level.map(med)
    m["residual"] = m["hr_mean"] - m["expected_hr"]
    m["level"] = level
    w = m.set_index("minute").resample(window)
    agg = pd.DataFrame({
        "mean_activity": w["activity"].mean(),
        "steps_sum": w["steps"].sum() if "steps" in m else w["activity"].count() * np.nan,
        "asleep_frac": w["is_asleep"].mean(),
        "expected_hr": w["expected_hr"].mean(),
        "residual_mean": w["residual"].mean(),
        "residual_max": w["residual"].max(),
        "residual_min": w["residual"].min(),
    })
    agg.index.name = "window_start"
    out = feat.merge(agg.reset_index(), on="window_start", how="left")
    return out


def daily_resting(ctx: ContextResult, min_minutes: int = 30) -> pd.DataFrame:
    """Per-day resting HR using the same rule as the headline value."""
    m = ctx.minutes
    if m.empty or "is_rest" not in m:
        return pd.DataFrame(columns=["date", "resting_hr"])
    r = m[m["is_rest"]]
    g = r.groupby(r["minute"].dt.date)["hr_mean"]
    out = pd.DataFrame({"resting_hr": g.quantile(0.25).round(1), "n": g.size()})
    out = out[out["n"] >= min_minutes].drop(columns="n")
    out.index.name = "date"
    return out.reset_index()
