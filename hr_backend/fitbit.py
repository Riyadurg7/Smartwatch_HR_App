"""
Loaders for Fitbit / Fitabase export files, and the minute-level merge that
lets the analysis use sleep and activity context.

Files understood (any subset):
  heartrate_seconds_merged.csv      Id, Time, Value               (bpm, ~5-15 s)
  minuteIntensitiesNarrow_merged    Id, ActivityMinute, Intensity (0 sedentary,
                                                                   1 light, 2 moderate,
                                                                   3 very active)
  minuteStepsNarrow_merged          Id, ActivityMinute, Steps
  minuteSleep_merged                Id, date, value, logId        (1 asleep,
                                                                   2 restless, 3 awake)
  dailyActivity_merged              Id, ActivityDate, TotalSteps, ... Calories

Multiple export periods can be passed together; overlapping rows are
de-duplicated.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

PathOrFrame = Union[str, pd.DataFrame]

FITBIT_TS = "%m/%d/%Y %I:%M:%S %p"
FITBIT_DATE = "%m/%d/%Y"

INTENSITY_LABELS = {0: "sedentary", 1: "light", 2: "moderate", 3: "vigorous"}
SLEEP_LABELS = {1: "asleep", 2: "restless", 3: "awake-in-bed"}


def _read(src: PathOrFrame) -> pd.DataFrame:
    return src.copy() if isinstance(src, pd.DataFrame) else pd.read_csv(src)


def _concat(sources: Iterable[PathOrFrame]) -> pd.DataFrame:
    frames = [_read(s) for s in sources if s is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _ts(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, format=FITBIT_TS, errors="coerce")
    if out.isna().mean() > 0.5:  # not the Fitbit format - let pandas infer
        out = pd.to_datetime(series, errors="coerce")
    return out


# --------------------------------------------------------------------------- #
# individual loaders -> tidy frames keyed on (Id, minute)
# --------------------------------------------------------------------------- #
def load_heartrate(sources: Iterable[PathOrFrame]) -> pd.DataFrame:
    """Raw second-level heart rate: Id, Time (str), Value. Duplicates dropped."""
    df = _concat(sources)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["Id", "Time"])
    return df.reset_index(drop=True)


def load_intensity(sources: Iterable[PathOrFrame]) -> pd.DataFrame:
    df = _concat(sources)
    if df.empty:
        return df
    df["minute"] = _ts(df["ActivityMinute"]).dt.floor("min")
    df = df.dropna(subset=["minute"]).drop_duplicates(subset=["Id", "minute"])
    return df[["Id", "minute", "Intensity"]].rename(columns={"Intensity": "intensity"})


def load_steps(sources: Iterable[PathOrFrame]) -> pd.DataFrame:
    df = _concat(sources)
    if df.empty:
        return df
    df["minute"] = _ts(df["ActivityMinute"]).dt.floor("min")
    df = df.dropna(subset=["minute"]).drop_duplicates(subset=["Id", "minute"])
    return df[["Id", "minute", "Steps"]].rename(columns={"Steps": "steps"})


def load_sleep(sources: Iterable[PathOrFrame]) -> pd.DataFrame:
    df = _concat(sources)
    if df.empty:
        return df
    # Sleep rows are stamped at hh:mm:30 - floor to the minute so they align.
    df["minute"] = _ts(df["date"]).dt.floor("min")
    df = df.dropna(subset=["minute"]).drop_duplicates(subset=["Id", "minute"])
    return df[["Id", "minute", "value"]].rename(columns={"value": "sleep_state"})


def load_daily(sources: Iterable[PathOrFrame]) -> pd.DataFrame:
    df = _concat(sources)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["ActivityDate"], format=FITBIT_DATE, errors="coerce").dt.date
    df = df.dropna(subset=["date"]).drop_duplicates(subset=["Id", "date"])
    return df


# --------------------------------------------------------------------------- #
# minute-level merge
# --------------------------------------------------------------------------- #
def heartrate_per_minute(hr: pd.DataFrame) -> pd.DataFrame:
    """Aggregate second-level HR into per-minute mean / std / min / max / count."""
    if hr.empty:
        return pd.DataFrame()
    d = hr.copy()
    d["ts"] = _ts(d["Time"])
    d["Value"] = pd.to_numeric(d["Value"], errors="coerce")
    d = d.dropna(subset=["ts", "Value"])
    d = d[(d["Value"] >= 25) & (d["Value"] <= 240)]
    d["minute"] = d["ts"].dt.floor("min")
    g = d.groupby(["Id", "minute"])["Value"]
    out = g.agg(hr_mean="mean", hr_std="std", hr_min="min", hr_max="max", hr_n="count").reset_index()
    out["hr_std"] = out["hr_std"].fillna(0.0)
    return out


def build_minute_table(
    hr: pd.DataFrame,
    intensity: Optional[pd.DataFrame] = None,
    steps: Optional[pd.DataFrame] = None,
    sleep: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    One row per (Id, minute) where heart rate exists, with activity and sleep
    context joined on. Missing context stays NaN (sleep NaN = not in a logged
    sleep period).
    """
    m = heartrate_per_minute(hr)
    if m.empty:
        return m
    for extra in (intensity, steps, sleep):
        if extra is not None and not extra.empty:
            m = m.merge(extra, on=["Id", "minute"], how="left")
    if "sleep_state" in m:
        m["is_asleep"] = m["sleep_state"].isin([1, 2])
    return m.sort_values(["Id", "minute"]).reset_index(drop=True)


def load_export_folders(folders: Iterable[str]) -> dict:
    """
    Convenience for a local Fitbit export: pass one or more 'Fitabase Data ...'
    folders, get back {'hr','intensity','steps','sleep','daily'} frames.
    """
    from pathlib import Path

    def files(name):
        return [str(p) for f in folders for p in Path(f).glob(name)]

    return {
        "hr": load_heartrate(files("heartrate_seconds_merged.csv")),
        "intensity": load_intensity(files("minuteIntensitiesNarrow_merged.csv")),
        "steps": load_steps(files("minuteStepsNarrow_merged.csv")),
        "sleep": load_sleep(files("minuteSleep_merged.csv")),
        "daily": load_daily(files("dailyActivity_merged.csv")),
    }


def classify_companion(df: pd.DataFrame) -> Optional[str]:
    """Identify an uploaded Fitbit companion file by its columns."""
    cols = {c.lower() for c in df.columns}
    if {"activityminute", "intensity"} <= cols:
        return "intensity"
    if {"activityminute", "steps"} <= cols:
        return "steps"
    if {"date", "value", "logid"} <= cols:
        return "sleep"
    if {"time", "value"} <= cols and "id" in cols:
        return "heartrate"
    return None


def read_companions(frames: Iterable[pd.DataFrame]) -> dict:
    """
    Sort uploaded companion frames into tidy context frames:
    {'intensity': df|None, 'steps': df|None, 'sleep': df|None, 'unknown': n}
    """
    buckets = {"intensity": [], "steps": [], "sleep": []}
    unknown = 0
    for f in frames:
        kind = classify_companion(f)
        if kind in buckets:
            buckets[kind].append(f)
        else:
            unknown += 1
    return {
        "intensity": load_intensity(buckets["intensity"]) if buckets["intensity"] else None,
        "steps": load_steps(buckets["steps"]) if buckets["steps"] else None,
        "sleep": load_sleep(buckets["sleep"]) if buckets["sleep"] else None,
        "unknown": unknown,
    }
