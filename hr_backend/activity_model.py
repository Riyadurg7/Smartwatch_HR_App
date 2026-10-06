"""
Supervised activity-level model (Objective 3, supervised path, real labels).

Task: from heart rate alone, predict each minute's activity level:
    0 = sedentary, 1 = light, 2 = active (Fitbit moderate + vigorous)
Labels come from Fitbit's minuteIntensitiesNarrow files, so they are real
device labels, not invented ones.

Why it matters for the app: most users upload only a heart-rate CSV. The
model lets the app still tell "high HR because moving" from "high HR at rest".

Generalisation: features are normalised within each recording (relative to that
person's own baseline), and evaluation is leave-participants-out, so reported
metrics reflect performance on people the model has never seen.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "activity_model.joblib"
META_PATH = MODEL_DIR / "activity_model.json"

CLASSES = ["sedentary", "light", "active"]

FEATURES = [
    "hr_mean", "hr_std", "hr_range",
    "rel", "ratio", "z",
    "c5_mean", "c15_mean", "c5_std", "c15_max", "c15_min",
    "rel_c5", "rel_c15", "d1", "slope5",
    "hour_sin", "hour_cos",
]


def make_features(minute_hr: pd.DataFrame) -> pd.DataFrame:
    """
    Build model features from per-minute HR (columns: minute, hr_mean, hr_std,
    hr_min, hr_max; optional Id). Rolling features use a full 1-minute grid so
    gaps are respected. Returns one row per input row (same order).
    """
    df = minute_hr.copy()
    df["_order"] = np.arange(len(df))
    keys = ["Id"] if "Id" in df.columns else []
    parts = []
    for _, g in (df.groupby(keys) if keys else [(None, df)]):
        g = g.sort_values("minute").set_index("minute")
        grid = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="min"))
        hr = grid["hr_mean"]
        base = hr.quantile(0.10)
        med, sd = hr.median(), hr.std() or 1.0
        f = pd.DataFrame(index=grid.index)
        f["hr_mean"] = hr
        f["hr_std"] = grid["hr_std"]
        f["hr_range"] = grid["hr_max"] - grid["hr_min"]
        f["rel"] = hr - base
        f["ratio"] = hr / base
        f["z"] = (hr - med) / sd
        f["c5_mean"] = hr.rolling(5, center=True, min_periods=1).mean()
        f["c15_mean"] = hr.rolling(15, center=True, min_periods=1).mean()
        f["c5_std"] = hr.rolling(5, center=True, min_periods=2).std()
        f["c15_max"] = hr.rolling(15, center=True, min_periods=1).max()
        f["c15_min"] = hr.rolling(15, center=True, min_periods=1).min()
        f["rel_c5"] = f["c5_mean"] - base
        f["rel_c15"] = f["c15_mean"] - base
        f["d1"] = hr.diff()
        f["slope5"] = (hr.shift(-2) - hr.shift(2)) / 4
        h = f.index.hour + f.index.minute / 60
        f["hour_sin"] = np.sin(2 * np.pi * h / 24)
        f["hour_cos"] = np.cos(2 * np.pi * h / 24)
        f = f.loc[g.index]
        f["_order"] = g["_order"].values
        parts.append(f)
    out = pd.concat(parts).sort_values("_order")
    return out[FEATURES].reset_index(drop=True)


def labels_from_intensity(intensity: pd.Series) -> pd.Series:
    return intensity.map({0: 0, 1: 1, 2: 2, 3: 2})


def _new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
        l2_regularization=1.0, class_weight="balanced", random_state=42,
    )


def evaluate_lopo(minute_table: pd.DataFrame, n_splits: int = 5) -> dict:
    """
    Leave-participants-out cross-validation (GroupKFold by Id).
    Reports accuracy, balanced accuracy, macro precision/recall/F1, macro
    one-vs-rest ROC-AUC, active-vs-rest ROC-AUC, confusion matrix, and two
    baselines (always-sedentary, and a heart-rate-threshold logistic model).
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 precision_recall_fscore_support, roc_auc_score,
                                 confusion_matrix)

    data = minute_table.dropna(subset=["intensity", "hr_mean"]).reset_index(drop=True)
    X = make_features(data)
    y = labels_from_intensity(data["intensity"]).astype(int).values
    groups = data["Id"].values

    oof_pred = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros((len(y), 3))
    base_pred = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = _new_model().fit(X.iloc[tr], y[tr])
        oof_proba[te] = m.predict_proba(X.iloc[te])
        oof_pred[te] = oof_proba[te].argmax(1)
        # baseline: logistic regression on relative HR only (a "threshold rule")
        lr = LogisticRegression(max_iter=500, class_weight="balanced")
        xr = X[["rel"]].fillna(0)
        lr.fit(xr.iloc[tr], y[tr])
        base_pred[te] = lr.predict(xr.iloc[te])

    p, r, f, _ = precision_recall_fscore_support(y, oof_pred, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y, oof_pred, average=None, zero_division=0)
    bp, br, bf, _ = precision_recall_fscore_support(y, base_pred, average="macro", zero_division=0)
    metrics = {
        "n_minutes": int(len(y)),
        "n_participants": int(len(np.unique(groups))),
        "class_counts": {CLASSES[i]: int((y == i).sum()) for i in range(3)},
        "accuracy": round(float(accuracy_score(y, oof_pred)), 3),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, oof_pred)), 3),
        "precision_macro": round(float(p), 3),
        "recall_macro": round(float(r), 3),
        "f1_macro": round(float(f), 3),
        "roc_auc_macro_ovr": round(float(roc_auc_score(y, oof_proba, multi_class="ovr", average="macro")), 3),
        "roc_auc_active_vs_rest": round(float(roc_auc_score((y == 2).astype(int), oof_proba[:, 2])), 3),
        "per_class": {CLASSES[i]: {"precision": round(float(pc[i]), 3), "recall": round(float(rc[i]), 3),
                                    "f1": round(float(fc[i]), 3), "support": int(sc[i])} for i in range(3)},
        "confusion_matrix": confusion_matrix(y, oof_pred).tolist(),
        "baseline_always_sedentary_accuracy": round(float((y == 0).mean()), 3),
        "baseline_hr_threshold": {"accuracy": round(float(accuracy_score(y, base_pred)), 3),
                                  "balanced_accuracy": round(float(balanced_accuracy_score(y, base_pred)), 3),
                                  "f1_macro": round(float(bf), 3)},
        "cv": f"GroupKFold({n_splits}) by participant - test participants never seen in training",
    }
    return metrics


def train_and_save(minute_table: pd.DataFrame, metrics: Optional[dict] = None) -> "ActivityModel":
    import joblib
    import sklearn

    data = minute_table.dropna(subset=["intensity", "hr_mean"]).reset_index(drop=True)
    X = make_features(data)
    y = labels_from_intensity(data["intensity"]).astype(int).values
    model = _new_model().fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    meta = {
        "task": "per-minute activity level from heart rate",
        "classes": CLASSES,
        "features": FEATURES,
        "algorithm": "HistGradientBoostingClassifier(class_weight='balanced')",
        "trained_on": "FitBit Fitness Tracker Data (both export periods), "
                      f"{data['Id'].nunique()} participants, {len(y):,} labelled minutes",
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation": metrics or {},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    return ActivityModel(model, meta)


class ActivityModel:
    def __init__(self, model, meta: dict):
        self.model, self.meta = model, meta

    @classmethod
    def load(cls) -> Optional["ActivityModel"]:
        """Load the bundled model; returns None (app degrades gracefully) on failure."""
        try:
            import joblib
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = joblib.load(MODEL_PATH)
            meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
            return cls(model, meta)
        except Exception:
            return None

    def predict(self, minute_hr: pd.DataFrame) -> pd.DataFrame:
        X = make_features(minute_hr)
        proba = self.model.predict_proba(X)
        return pd.DataFrame({
            "activity": proba.argmax(1),
            "p_sedentary": proba[:, 0], "p_light": proba[:, 1], "p_active": proba[:, 2],
        })
