"""
Automated interpretation engine.

Objective 4: compute resting HR, max HR, variability stats, tachycardia /
bradycardia alerts, trend summaries and personalized plain-text explanations.

Every output is informational only. A non-clinical disclaimer is attached to
the result and MUST be shown in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .features import HRStats

DISCLAIMER = (
    "This analysis is generated automatically from heart-rate data for "
    "informational and educational purposes only. It is NOT a medical device, "
    "NOT a diagnosis, and must not be used to make health decisions. "
    "Consult a qualified clinician for any medical concern."
)

# Resting-state thresholds (applied to quiet/resting heart rate).
BRADYCARDIA_BPM = 60
TACHYCARDIA_BPM = 100


@dataclass
class Alert:
    level: str          # "info" | "attention"
    code: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Interpretation:
    stats: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    summary_text: str = ""
    trend_text: str = ""
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _num(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.1f}"


def interpret(
    stats: HRStats,
    daily: Optional[pd.DataFrame] = None,
    circadian: Optional[pd.DataFrame] = None,
    anomaly_count: int = 0,
    n_windows: int = 0,
    age: Optional[int] = None,
    ctx=None,
    model_meta: Optional[dict] = None,
) -> Interpretation:
    """
    Produce alerts + narrative from computed statistics.

    age (optional) enables an age-predicted max-HR comparison (220 - age).
    """
    alerts: list[Alert] = []

    # --- resting HR interpretation (AWAKE resting HR, not the sleep low) ---
    rhr = stats.resting_hr
    if not np.isnan(rhr):
        if rhr > TACHYCARDIA_BPM:
            alerts.append(Alert(
                "attention", "resting_tachy",
                f"Awake resting heart rate looks elevated (~{_fmt(rhr)} bpm, above "
                f"{TACHYCARDIA_BPM}). Caffeine, stress, dehydration, illness or poor "
                "sleep can raise it; if it stays high at rest, it is worth reviewing."
            ))
        elif rhr < 50:
            # Only genuinely low AWAKE resting rates are flagged, and gently.
            alerts.append(Alert(
                "attention", "resting_low",
                f"Awake resting heart rate is quite low (~{_fmt(rhr)} bpm). This is common "
                "and often healthy in fit or athletic people, but a low resting rate "
                "together with dizziness or fatigue is worth discussing with a clinician."
            ))
        elif rhr < BRADYCARDIA_BPM:
            alerts.append(Alert(
                "info", "resting_low_normal",
                f"Awake resting heart rate (~{_fmt(rhr)} bpm) is on the lower side of "
                "normal - typical of regular exercisers."
            ))
        else:
            alerts.append(Alert(
                "info", "resting_normal",
                f"Awake resting heart rate (~{_fmt(rhr)} bpm) is within the typical "
                "60-100 bpm range."
            ))

    # Overnight low is reported as context, explicitly NOT bradycardia.
    if not np.isnan(stats.overnight_low_hr):
        alerts.append(Alert(
            "info", "overnight_low",
            f"Lowest sustained overnight heart rate was ~{_fmt(stats.overnight_low_hr)} bpm. "
            "Heart rate naturally dips during deep sleep, so a low overnight value is "
            "expected and is not the same as resting bradycardia."
        ))

    # --- sleep & activity context ---
    activity_text = ""
    if ctx is not None:
        if ctx.intensity_source == "estimated":
            acc = (model_meta or {}).get("evaluation", {}).get("accuracy")
            acc_txt = f" (about {acc*100:.0f}% accurate on people it was not trained on)" if acc else ""
            alerts.append(Alert(
                "info", "activity_estimated",
                "No activity file was provided, so activity levels were estimated from "
                f"heart rate by the trained model{acc_txt}. Upload Fitbit intensity/steps "
                "files for exact activity context."
            ))
        if ctx.sleep_source == "observed" and not np.isnan(ctx.sleep_hr_mean):
            alerts.append(Alert(
                "info", "sleep_hr",
                f"During logged sleep ({ctx.sleep_minutes/60:.0f} h in total) heart rate averaged "
                f"~{_fmt(ctx.sleep_hr_mean)} bpm."
            ))
        eps = ctx.elevated_rest_episodes or []
        if eps:
            total = sum(e["minutes"] for e in eps)
            lvl = "attention" if (len(eps) >= 3 or total >= 60) else "info"
            alerts.append(Alert(
                lvl, "elevated_at_rest",
                f"{len(eps)} period(s) ({total} min in total) where heart rate stayed above "
                f"{max(100, _num(ctx.resting_hr) + 30):.0f} bpm while you were still and awake. "
                "Stress, caffeine, illness or a loose watch strap can cause this; if it "
                "happens often or with symptoms, it is worth discussing with a clinician."
            ))
        rows = {r["activity"]: r for r in (ctx.hr_by_activity or [])}
        if rows:
            bits = [f"{rows[k]['pct']:.0f}% {k}" for k in ("sedentary", "light", "active", "asleep") if k in rows]
            act = rows.get("active", {})
            sed = rows.get("sedentary", {})
            activity_text = (
                "Of the recorded time: " + ", ".join(bits) + "."
                + (f" Typical heart rate was ~{_fmt(sed.get('median_hr'))} bpm when sedentary and "
                   f"~{_fmt(act.get('median_hr'))} bpm when active." if act.get("minutes") else "")
            )

    # --- max HR / age comparison ---
    if age is not None and age > 0 and not np.isnan(stats.max_hr):
        predicted = 220 - age
        pct = 100 * stats.max_hr / predicted
        alerts.append(Alert(
            "info", "max_hr_age",
            f"Peak heart rate reached ~{_fmt(stats.max_hr)} bpm, about {pct:.0f}% of the "
            f"age-predicted maximum ({predicted} bpm for age {age})."
        ))

    # --- variability note ---
    if not np.isnan(stats.hrv_proxy_ms):
        alerts.append(Alert(
            "info", "hrv_proxy",
            f"Heart-rate variability proxy is ~{_fmt(stats.hrv_proxy_ms)} ms "
            "(derived from BPM data, not true inter-beat HRV - use for trend tracking only)."
        ))

    # --- anomalies ---
    if n_windows > 0 and anomaly_count > 0:
        pct = 100 * anomaly_count / n_windows
        lvl = "attention" if pct > 10 else "info"
        alerts.append(Alert(
            lvl, "anomalies",
            f"{anomaly_count} of {n_windows} time windows ({pct:.0f}%) were flagged as "
            "unusual relative to the rest of the recording"
            + (" after accounting for activity and sleep" if ctx is not None and ctx.available else "")
            + ". These are statistical outliers (stress, unusual effort or sensor "
            "artifacts), not necessarily problems."
        ))

    summary_text = _build_summary(stats)
    if activity_text:
        summary_text += " " + activity_text
    trend_text = _build_trend(daily, circadian)

    return Interpretation(
        stats=stats.to_dict(),
        alerts=[a.to_dict() for a in alerts],
        summary_text=summary_text,
        trend_text=trend_text,
    )


def _build_summary(stats: HRStats) -> str:
    parts = [
        f"Across this recording the average heart rate was {_fmt(stats.mean_hr)} bpm,",
        f"ranging from about {_fmt(stats.p05)} to {_fmt(stats.p95)} bpm "
        "(5th-95th percentile).",
        f"The estimated awake resting heart rate is {_fmt(stats.resting_hr)} bpm",
        f"and the peak reached {_fmt(stats.max_hr)} bpm.",
    ]
    if not np.isnan(stats.overnight_low_hr):
        parts.append(f"Overnight heart rate dipped to about {_fmt(stats.overnight_low_hr)} bpm.")
    return " ".join(parts)


def _build_trend(daily: Optional[pd.DataFrame], circadian: Optional[pd.DataFrame]) -> str:
    bits = []
    dd = daily.dropna(subset=["resting_hr"]) if daily is not None and not daily.empty else None
    if dd is not None and len(dd) >= 2:
        # compare the average of the first and last third of days, so one odd
        # day does not create a "trend"; under 2 bpm counts as stable
        k = max(1, len(dd) // 3)
        early = round(float(dd["resting_hr"].iloc[:k].mean()), 1)
        late = round(float(dd["resting_hr"].iloc[-k:].mean()), 1)
        delta = late - early
        direction = "risen" if delta > 2 else "fallen" if delta < -2 else "stayed about the same"
        bits.append(
            f"Over {len(dd)} days, resting heart rate has {direction} "
            f"(about {_fmt(early)} bpm at the start, {_fmt(late)} bpm at the end)."
        )
    if circadian is not None and not circadian.empty and circadian["mean_hr"].notna().any():
        lo = circadian.loc[circadian["mean_hr"].idxmin()]
        hi = circadian.loc[circadian["mean_hr"].idxmax()]
        bits.append(
            f"Heart rate is typically lowest around {int(lo['hour']):02d}:00 "
            f"(~{_fmt(lo['mean_hr'])} bpm) and highest around {int(hi['hour']):02d}:00 "
            f"(~{_fmt(hi['mean_hr'])} bpm), consistent with a normal circadian pattern."
        )
    return " ".join(bits) if bits else "Not enough data for a multi-day trend summary."
