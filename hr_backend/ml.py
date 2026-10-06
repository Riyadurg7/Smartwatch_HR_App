"""
Machine-learning module.

Objective 3: anomaly detection (Isolation Forest / Local Outlier Factor),
clustering (KMeans), and an optional supervised classifier with standard
metrics (accuracy, precision, recall, F1, ROC-AUC) for when labelled data
is available.

All functions operate on the windowed feature matrix from
features.windowed_feature_matrix().
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Shape features used for clustering (behaviour regimes).
_CLUSTER_COLS = ["mean", "std", "min", "max", "range", "slope"]

# Anomaly features when no activity context exists: shape + largest sudden jump.
_FEATURE_COLS = ["mean", "std", "min", "max", "range", "slope", "max_jump"]

# Context-aware anomaly features: how far HR is from this person's normal for the
# activity level / sleep of each window (mean, highest, lowest), plus the largest
# sudden jump. Exercise with appropriately high HR is NOT unusual under these;
# high HR while still, or abrupt glitches, are. Variability features (std, range,
# slope) are deliberately excluded: they rise with ordinary movement and made the
# detector flag workouts (see VALIDATION.md, section 4).
CONTEXT_FEATURE_COLS = ["residual_mean", "residual_max", "residual_min", "max_jump"]


def _matrix(feat: pd.DataFrame, cols=None) -> tuple[np.ndarray, list]:
    cols = [c for c in (cols or _FEATURE_COLS) if c in feat.columns]
    X = feat[cols].fillna(0.0).to_numpy()
    return X, cols


@dataclass
class AnomalyResult:
    method: str = "isolation_forest"
    contamination: float = 0.05
    features_used: list = field(default_factory=list)
    context_aware: bool = False
    n_windows: int = 0
    n_anomalies: int = 0
    table: pd.DataFrame = field(default_factory=pd.DataFrame)  # windows + is_anomaly + score

    def to_dict(self) -> dict:
        d = asdict(self)
        d["table"] = self.table.to_dict(orient="records")
        return d


def detect_anomalies(
    feat: pd.DataFrame,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    random_state: int = 42,
    feature_cols: Optional[list] = None,
) -> AnomalyResult:
    """
    Flag anomalous time windows.

    method : 'isolation_forest' or 'lof'
    contamination : expected fraction of anomalies (configurable from the UI)

    Returns AnomalyResult with a `table` = feat + ['anomaly_score', 'is_anomaly'].
    Higher anomaly_score = more anomalous (sign-normalized across methods).
    """
    if feat.empty or len(feat) < 5:
        return AnomalyResult(method=method, contamination=contamination,
                             n_windows=len(feat), table=feat.copy())

    X, cols = _matrix(feat, feature_cols)
    Xs = StandardScaler().fit_transform(X)
    out = feat.copy()

    if method == "lof":
        n_neighbors = min(20, max(2, len(feat) - 1))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        labels = lof.fit_predict(Xs)                 # -1 anomaly, 1 normal
        scores = -lof.negative_outlier_factor_       # higher = more anomalous
    else:
        method = "isolation_forest"
        iso = IsolationForest(
            contamination=contamination, random_state=random_state, n_estimators=200
        )
        labels = iso.fit_predict(Xs)                 # -1 anomaly, 1 normal
        scores = -iso.score_samples(Xs)              # higher = more anomalous

    out["anomaly_score"] = np.round(scores, 4)
    out["is_anomaly"] = labels == -1

    return AnomalyResult(
        method=method,
        contamination=contamination,
        features_used=cols,
        context_aware=any(c.startswith("residual") for c in cols),
        n_windows=len(feat),
        n_anomalies=int(out["is_anomaly"].sum()),
        table=out,
    )


@dataclass
class ClusterResult:
    k: int = 0
    labels: list = field(default_factory=list)
    centers: list = field(default_factory=list)
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    profile: pd.DataFrame = field(default_factory=pd.DataFrame)  # mean feats per cluster

    def to_dict(self) -> dict:
        d = asdict(self)
        d["table"] = self.table.to_dict(orient="records")
        d["profile"] = self.profile.to_dict(orient="records")
        return d


def cluster_patterns(
    feat: pd.DataFrame,
    k: Optional[int] = None,
    random_state: int = 42,
) -> ClusterResult:
    """
    KMeans clustering of HR windows into behavioural regimes
    (e.g. rest / light activity / exertion). Auto-picks k in [2,5] by silhouette
    when k is None.
    """
    if feat.empty or len(feat) < 4:
        return ClusterResult(table=feat.copy())

    X, cols = _matrix(feat, _CLUSTER_COLS)
    Xs = StandardScaler().fit_transform(X)

    if k is None:
        k = _best_k(Xs, k_range=range(2, min(6, len(feat))))
    k = max(1, min(k, len(feat)))

    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(Xs)

    out = feat.copy()
    out["cluster"] = labels

    # order clusters by mean HR so labels are interpretable (0 = lowest HR)
    order = out.groupby("cluster")["mean"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    out["cluster"] = out["cluster"].map(remap)

    profile = (
        out.groupby("cluster")[cols + ["n"]]
        .mean()
        .round(1)
        .reset_index()
    )
    return ClusterResult(
        k=k,
        labels=out["cluster"].tolist(),
        centers=km.cluster_centers_.tolist(),
        table=out,
        profile=profile,
    )


def _best_k(Xs: np.ndarray, k_range) -> int:
    from sklearn.metrics import silhouette_score
    best_k, best_s = 2, -1.0
    for k in k_range:
        if k >= len(Xs):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        if len(set(km.labels_)) < 2:
            continue
        s = silhouette_score(Xs, km.labels_)
        if s > best_s:
            best_k, best_s = k, s
    return best_k


@dataclass
class ClassifierResult:
    trained: bool = False
    metrics: dict = field(default_factory=dict)
    confusion: list = field(default_factory=list)
    feature_importance: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def train_supervised(
    feat: pd.DataFrame,
    label_col: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ClassifierResult:
    """
    Objective 3 (supervised path): train a classifier when a labelled column is
    present (e.g. activity level, normal/abnormal). Reports accuracy, precision,
    recall, F1 and ROC-AUC. This is a scaffold the project can point at any
    labelled dataset; it is skipped automatically when no labels exist.
    """
    if label_col not in feat.columns:
        return ClassifierResult(note=f"No label column '{label_col}' - supervised step skipped.")

    data = feat.dropna(subset=[label_col])
    X, cols = _matrix(data)
    y = data[label_col].to_numpy()

    if len(np.unique(y)) < 2 or len(data) < 10:
        return ClassifierResult(note="Not enough labelled samples/classes to train.")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    avg = "binary" if len(np.unique(y)) == 2 else "macro"
    metrics = {
        "accuracy": round(float(accuracy_score(yte, pred)), 3),
        "precision": round(float(precision_score(yte, pred, average=avg, zero_division=0)), 3),
        "recall": round(float(recall_score(yte, pred, average=avg, zero_division=0)), 3),
        "f1": round(float(f1_score(yte, pred, average=avg, zero_division=0)), 3),
    }
    # ROC-AUC (binary only, needs probabilities)
    try:
        if len(np.unique(y)) == 2:
            proba = clf.predict_proba(Xte)[:, 1]
            metrics["roc_auc"] = round(float(roc_auc_score(yte, proba)), 3)
    except Exception:
        pass

    return ClassifierResult(
        trained=True,
        metrics=metrics,
        confusion=confusion_matrix(yte, pred).tolist(),
        feature_importance={c: round(float(i), 3) for c, i in zip(cols, clf.feature_importances_)},
        note=f"Trained RandomForest on {len(data)} labelled windows.",
    )
