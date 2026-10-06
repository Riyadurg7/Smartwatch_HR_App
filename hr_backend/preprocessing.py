"""
Preprocessing pipeline for smartwatch heart-rate data.

Objective 1: missing-value handling, timestamp parsing, duplicate removal,
normalization, signal smoothing and quality assessment -> analysis-ready data.

The main entrypoint is `preprocess()`. It accepts a raw DataFrame (or a CSV
path via `load_csv`) and returns a cleaned, regularly-sampled DataFrame plus a
quality report.

Supported input formats
-----------------------
- Fitbit `heartrate_seconds_merged.csv`  (columns: Id, Time, Value)
- Generic CSV with any timestamp column + any heart-rate/BPM column
  (auto-detected; can also be passed explicitly).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

# Physiologically plausible instantaneous heart-rate bounds (bpm).
# Values outside this range are treated as sensor artifacts.
HR_MIN_PLAUSIBLE = 25
HR_MAX_PLAUSIBLE = 240

# Candidate column names (lower-cased) used for auto-detection.
_TIME_HINTS = ("time", "timestamp", "datetime", "date", "activityhour", "activityminute")
_HR_HINTS = ("value", "heartrate", "heart_rate", "hr", "bpm", "beatsperminute", "pulse")


@dataclass
class QualityReport:
    """Summary of input data quality, surfaced to the UI."""
    n_raw_rows: int = 0
    n_after_clean: int = 0
    n_duplicates_removed: int = 0
    n_implausible_removed: int = 0
    n_missing_filled: int = 0
    pct_missing_raw: float = 0.0
    start: Optional[str] = None
    end: Optional[str] = None
    duration_hours: float = 0.0
    median_sampling_seconds: Optional[float] = None
    resample_rule: Optional[str] = None
    coverage_pct: float = 0.0            # % of resampled bins that had real data
    quality_score: float = 0.0          # 0-100 heuristic
    quality_label: str = "unknown"
    user_id: Optional[str] = None       # participant analysed (multi-user files)
    n_users_in_file: int = 1
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_csv(path: str, **read_kwargs) -> pd.DataFrame:
    """Load a CSV from disk. Thin wrapper so the frontend has one import point."""
    return pd.read_csv(path, **read_kwargs)


def _detect_column(df: pd.DataFrame, hints: tuple) -> Optional[str]:
    """Find the first column whose lower-cased name matches a hint."""
    lowered = {c.lower().strip(): c for c in df.columns}
    # exact match first
    for h in hints:
        if h in lowered:
            return lowered[h]
    # substring match second
    for col_lower, original in lowered.items():
        if any(h in col_lower for h in hints):
            return original
    return None


def _coerce_hr_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# Common timestamp formats, tried in order before falling back to the slow,
# element-wise dateutil parser (which also emits a pandas warning).
_TS_FORMATS = (
    "%m/%d/%Y %I:%M:%S %p",   # Fitbit export, e.g. 4/12/2016 7:21:00 AM
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def _parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse timestamps, trying known formats first for speed and to avoid warnings."""
    s = series.astype(str).str.strip()
    for fmt in _TS_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return parsed
    # Numeric epoch seconds/millis?
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.9:
        unit = "ms" if num.max() > 1e11 else "s"
        return pd.to_datetime(num, unit=unit, errors="coerce")
    # Fallback: let pandas infer (handles ISO and odd formats).
    return pd.to_datetime(series, errors="coerce")


def standardize(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    hr_col: Optional[str] = None,
    id_col: Optional[str] = None,
    user_id=None,
) -> pd.DataFrame:
    """
    Reduce an arbitrary input frame to a canonical two-column frame:
    ['timestamp' (datetime64), 'hr' (float)], sorted by time.

    If an id column exists and `user_id` is given, filter to that user; otherwise
    if multiple users are present, the single most-represented user is kept and a
    warning is attached later (multi-user files are not merged, since that would
    mix physiologies).
    """
    df = df.copy()

    time_col = time_col or _detect_column(df, _TIME_HINTS)
    hr_col = hr_col or _detect_column(df, _HR_HINTS)
    if time_col is None or hr_col is None:
        raise ValueError(
            "Could not auto-detect timestamp and/or heart-rate columns. "
            f"Columns seen: {list(df.columns)}. "
            "Pass time_col= and hr_col= explicitly."
        )

    id_col = id_col or _detect_column(df, ("id", "userid", "user_id", "subject"))
    chosen, n_users = None, 1
    if id_col is not None and id_col in df.columns:
        n_users = int(df[id_col].nunique())
        if user_id is not None:
            df = df[df[id_col] == user_id]
            chosen = user_id
        elif n_users > 1:
            chosen = df[id_col].value_counts().idxmax()
            df = df[df[id_col] == chosen]
        elif n_users == 1:
            chosen = df[id_col].iloc[0]

    out = pd.DataFrame(
        {
            "timestamp": _parse_timestamps(df[time_col]),
            "hr": _coerce_hr_numeric(df[hr_col]),
        }
    )
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # remember which participant was analysed (read immediately by the pipeline)
    out.attrs["user_id"] = None if chosen is None else (chosen.item() if hasattr(chosen, "item") else chosen)
    out.attrs["n_users"] = n_users
    return out


# Standard grid intervals (seconds -> pandas offset alias). We snap to the
# smallest standard interval that is >= the native sampling interval, so the
# regular grid never has more bins than there are real samples (which would
# create artificial gaps and deflate coverage).
_STD_BINS = [
    (1, "1s"), (5, "5s"), (10, "10s"), (15, "15s"), (30, "30s"),
    (60, "1min"), (300, "5min"), (900, "15min"), (1800, "30min"), (3600, "1h"),
]


def _infer_resample_rule(median_dt_seconds: float) -> str:
    """Pick a regular grid >= the native sampling interval."""
    for secs, alias in _STD_BINS:
        if median_dt_seconds <= secs:
            return alias
    return "1h"


def preprocess(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    hr_col: Optional[str] = None,
    id_col: Optional[str] = None,
    user_id=None,
    resample_rule: Optional[str] = None,
    smoothing_window: int = 5,
    max_gap_fill: int = 6,
) -> tuple[pd.DataFrame, QualityReport]:
    """
    Full cleaning pipeline.

    Parameters
    ----------
    resample_rule : pandas offset alias (e.g. '1min'). Auto-inferred if None.
    smoothing_window : window (in resampled samples) for rolling median+mean smoothing.
    max_gap_fill : max consecutive missing resampled bins to interpolate across.
                   Longer gaps are left as NaN (and excluded from coverage).

    Returns
    -------
    (clean_df, report)
        clean_df columns:
          timestamp, hr (clipped+deduped, native grid),
          hr_resampled (regular grid), hr_smooth (smoothed), is_filled (bool)
        The regular-grid series is indexed on `timestamp`.
    """
    report = QualityReport()
    canon = standardize(df, time_col, hr_col, id_col, user_id)
    report.n_raw_rows = len(canon)
    if report.n_raw_rows == 0:
        raise ValueError("No valid rows after timestamp parsing.")

    report.pct_missing_raw = float(canon["hr"].isna().mean() * 100)

    # --- duplicate timestamps: average the values ---
    before = len(canon)
    canon = (
        canon.groupby("timestamp", as_index=False)["hr"].mean().sort_values("timestamp")
    )
    report.n_duplicates_removed = before - len(canon)

    # --- implausible values -> NaN ---
    implausible = ~canon["hr"].between(HR_MIN_PLAUSIBLE, HR_MAX_PLAUSIBLE)
    report.n_implausible_removed = int(implausible.sum())
    canon.loc[implausible, "hr"] = np.nan

    # --- native sampling interval ---
    dt = canon["timestamp"].diff().dt.total_seconds().dropna()
    median_dt = float(dt.median()) if len(dt) else None
    report.median_sampling_seconds = median_dt

    rule = resample_rule or (_infer_resample_rule(median_dt) if median_dt else "1min")
    report.resample_rule = rule

    # --- resample onto regular grid (mean within each bin) ---
    s = canon.set_index("timestamp")["hr"]
    resampled = s.resample(rule).mean()

    n_bins = len(resampled)
    real_mask = resampled.notna()
    n_real = int(real_mask.sum())

    # interpolate only across short gaps
    filled = resampled.interpolate(method="time", limit=max_gap_fill, limit_area="inside")
    is_filled = resampled.isna() & filled.notna()
    report.n_missing_filled = int(is_filled.sum())

    # --- smoothing: rolling median (de-spike) then rolling mean (smooth) ---
    w = max(1, int(smoothing_window))
    smooth = (
        filled.rolling(w, center=True, min_periods=1).median()
        .rolling(w, center=True, min_periods=1).mean()
    )

    clean = pd.DataFrame(
        {
            "timestamp": resampled.index,
            "hr_resampled": filled.values,
            "hr_smooth": smooth.values,
            "is_filled": is_filled.values,
        }
    )

    # --- report fields ---
    report.n_after_clean = int(filled.notna().sum())
    report.start = str(resampled.index.min())
    report.end = str(resampled.index.max())
    span = (resampled.index.max() - resampled.index.min()).total_seconds()
    report.duration_hours = round(span / 3600, 2) if span else 0.0
    report.coverage_pct = round(100 * n_real / n_bins, 1) if n_bins else 0.0

    # --- heuristic quality score ---
    report.quality_score, report.quality_label, report.warnings = _score_quality(report)

    return clean, report


def _score_quality(r: QualityReport) -> tuple[float, str, list]:
    """Heuristic 0-100 data-quality score with human-readable warnings."""
    score = 100.0
    warnings: list = []

    if r.coverage_pct < 90:
        score -= (90 - r.coverage_pct) * 0.5
        warnings.append(f"Coverage is {r.coverage_pct:.0f}% - there are gaps in the recording.")
    if r.pct_missing_raw > 5:
        score -= min(20, r.pct_missing_raw)
        warnings.append(f"{r.pct_missing_raw:.0f}% of raw values were missing.")
    if r.n_implausible_removed > 0:
        frac = 100 * r.n_implausible_removed / max(1, r.n_raw_rows)
        score -= min(15, frac)
        if frac > 1:
            warnings.append(
                f"{r.n_implausible_removed} implausible readings "
                f"(<{HR_MIN_PLAUSIBLE} or >{HR_MAX_PLAUSIBLE} bpm) were removed."
            )
    if r.duration_hours < 1:
        score -= 15
        warnings.append("Recording is under 1 hour - trends may be unreliable.")

    score = float(max(0, min(100, round(score, 1))))
    label = (
        "excellent" if score >= 85
        else "good" if score >= 70
        else "fair" if score >= 50
        else "poor"
    )
    return score, label, warnings
