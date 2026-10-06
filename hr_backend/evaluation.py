"""
Honest evaluation of the anomaly detector (Objective 3 metrics).

The public Fitbit dataset has NO ground-truth anomaly labels, so we cannot
report real clinical detection accuracy. Instead we run a controlled,
semi-synthetic benchmark: take a real participant's cleaned signal, inject a
known number of anomalies (spikes, dropouts, sustained abnormal stretches) at
KNOWN locations, then measure how well the unsupervised detector recovers them.

This gives defensible precision / recall / F1 / ROC-AUC numbers while being
explicit that the positives are injected, not clinically confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
)

from . import features, ml


@dataclass
class EvalResult:
    n_windows: int = 0
    n_injected_windows: int = 0
    precision: float = np.nan
    recall: float = np.nan
    f1: float = np.nan
    roc_auc: float = np.nan
    pr_auc: float = np.nan
    method: str = ""
    contamination: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def inject_anomalies(
    clean: pd.DataFrame,
    n_spikes: int = 15,
    n_dropouts: int = 10,
    n_sustained: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Inject known anomalies into a cleaned signal.

    Returns (modified_clean, anomaly_time_mask) where anomaly_time_mask marks
    the timestamps that were tampered with.
    """
    rng = np.random.default_rng(seed)
    df = clean.copy().reset_index(drop=True)
    hr = df["hr_smooth"].to_numpy(dtype=float).copy()
    hr_rs = df["hr_resampled"].to_numpy(dtype=float).copy()
    n = len(df)
    mask = np.zeros(n, dtype=bool)
    valid = np.where(~np.isnan(hr))[0]
    if len(valid) < 50:
        return df, mask

    # point spikes (sudden very high / very low)
    for _ in range(n_spikes):
        i = int(rng.choice(valid))
        hr[i] = hr_rs[i] = rng.choice([rng.uniform(170, 210), rng.uniform(30, 42)])
        mask[i] = True

    # dropouts (physiologically implausible flatline / zero-ish)
    for _ in range(n_dropouts):
        i = int(rng.choice(valid))
        hr[i] = hr_rs[i] = rng.uniform(33, 40)
        mask[i] = True

    # sustained abnormal stretches (e.g. 10-min run of elevated HR at rest)
    for _ in range(n_sustained):
        start = int(rng.choice(valid))
        length = int(rng.integers(20, 60))
        end = min(n, start + length)
        bump = rng.uniform(45, 75)
        hr[start:end] = np.clip(hr[start:end] + bump, 35, 220)
        hr_rs[start:end] = hr[start:end]
        mask[start:end] = True

    df["hr_smooth"] = hr
    df["hr_resampled"] = hr_rs
    return df, mask


def evaluate_anomaly_detection(
    clean: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    window: str = "10min",
    seed: int = 0,
    context_frames: Optional[dict] = None,
    user_id=None,
    activity_model=None,
) -> EvalResult:
    """
    Inject known anomalies, run detection, and score window-level recovery.

    A window counts as a true anomaly if it overlaps any injected anomalous
    sample. We report precision/recall/F1 on the binary flags and ROC-AUC /
    PR-AUC on the continuous anomaly score.
    """
    tampered, time_mask = inject_anomalies(clean, seed=seed)
    if time_mask.sum() == 0:
        return EvalResult(note="Signal too short to inject anomalies.")

    # Label each analysis window as anomalous if it contains an injected sample.
    tampered = tampered.copy()
    tampered["_injected"] = time_mask
    feat = features.windowed_feature_matrix(tampered, window=window)
    if feat.empty:
        return EvalResult(note="No feature windows produced.")

    # Map injected flags onto windows by time.
    inj = tampered.set_index("timestamp")["_injected"]
    win_delta = pd.Timedelta(window)
    y_true = []
    for ws in feat["window_start"]:
        seg = inj.loc[ws: ws + win_delta]
        y_true.append(bool(seg.any()))
    y_true = np.array(y_true, dtype=int)

    cols = None
    if context_frames is not None or activity_model is not None:
        # context-aware mode: rebuild minute HR from the TAMPERED signal, then
        # join activity/sleep context and score on residual features
        from . import context as _ctx
        cf = context_frames or {}
        canon = tampered.rename(columns={"hr_resampled": "hr"})[["timestamp", "hr"]]
        mhr = _ctx.minute_hr_from_clean(canon)
        c = _ctx.build_context(mhr, cf.get("intensity"), cf.get("steps"), cf.get("sleep"),
                               user_id=user_id, activity_model=activity_model)
        feat = _ctx.add_window_context(feat, c, window)
        if "residual_mean" in feat:
            cols = ml.CONTEXT_FEATURE_COLS
    res = ml.detect_anomalies(feat, method=method, contamination=contamination, feature_cols=cols)
    tbl = res.table
    y_pred = tbl["is_anomaly"].to_numpy().astype(int)
    y_score = tbl["anomaly_score"].to_numpy()

    out = EvalResult(
        n_windows=len(feat),
        n_injected_windows=int(y_true.sum()),
        method=res.method,
        contamination=contamination,
        precision=round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        recall=round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        note="Positives are injected anomalies (no clinical ground truth exists in this dataset).",
    )
    try:
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            out.roc_auc = round(float(roc_auc_score(y_true, y_score)), 3)
            out.pr_auc = round(float(average_precision_score(y_true, y_score)), 3)
    except Exception:
        pass
    return out
