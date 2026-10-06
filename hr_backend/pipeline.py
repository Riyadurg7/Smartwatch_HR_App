"""
Pipeline orchestrator - the single entrypoint the Streamlit frontend calls.

    from hr_backend import analyze
    result = analyze(hr_df)                                   # heart rate only
    result = analyze(hr_df, intensity_df=..., steps_df=..., sleep_df=...)  # + context

Context frames come from hr_backend.fitbit loaders (load_intensity, load_steps,
load_sleep) or fitbit.read_companion() for uploaded files. When no intensity
file is given, the bundled supervised activity model estimates activity level
from heart rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import preprocessing, features, ml, interpretation, context
from .activity_model import ActivityModel

_MODEL_CACHE: dict = {}


def get_activity_model() -> Optional[ActivityModel]:
    if "m" not in _MODEL_CACHE:
        _MODEL_CACHE["m"] = ActivityModel.load()
    return _MODEL_CACHE["m"]


@dataclass
class AnalysisResult:
    # cleaned signals (regular grid)
    clean: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality: dict = field(default_factory=dict)

    # statistics & tables for charts
    stats: dict = field(default_factory=dict)
    daily: pd.DataFrame = field(default_factory=pd.DataFrame)
    circadian: pd.DataFrame = field(default_factory=pd.DataFrame)
    peaks: pd.DataFrame = field(default_factory=pd.DataFrame)

    # sleep / activity context
    context: dict = field(default_factory=dict)              # ContextResult.to_dict()
    minutes: pd.DataFrame = field(default_factory=pd.DataFrame)  # per-minute HR + activity + sleep

    # ML
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    anomalies: dict = field(default_factory=dict)
    clusters: dict = field(default_factory=dict)
    classifier: Optional[dict] = None
    activity_model: dict = field(default_factory=dict)       # model metadata + metrics

    # narrative
    interpretation: dict = field(default_factory=dict)

    ok: bool = True
    error: Optional[str] = None


def analyze(
    df: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
    hr_col: Optional[str] = None,
    id_col: Optional[str] = None,
    user_id=None,
    intensity_df: Optional[pd.DataFrame] = None,
    steps_df: Optional[pd.DataFrame] = None,
    sleep_df: Optional[pd.DataFrame] = None,
    use_activity_model: bool = True,
    context_aware_anomalies: bool = True,
    resample_rule: Optional[str] = None,
    smoothing_window: int = 5,
    window: str = "10min",
    anomaly_method: str = "isolation_forest",
    contamination: float = 0.05,
    n_clusters: Optional[int] = None,
    label_col: Optional[str] = None,
    age: Optional[int] = None,
) -> AnalysisResult:
    """
    Run the full analysis. Keyword args map to UI controls (Objective 5).
    Failures are captured in result.error so the frontend never crashes.
    """
    try:
        # 1. standardise + preprocess (Objective 1)
        canon = preprocessing.standardize(df, time_col, hr_col, id_col, user_id)
        uid = canon.attrs.get("user_id")
        n_users = canon.attrs.get("n_users", 1)
        clean, qreport = preprocessing.preprocess(
            canon, time_col="timestamp", hr_col="hr",
            resample_rule=resample_rule, smoothing_window=smoothing_window,
        )
        qreport.user_id = None if uid is None else str(uid)
        qreport.n_users_in_file = int(n_users)
        if n_users > 1:
            qreport.warnings.append(
                f"File contains {n_users} participants; analysing the one with the most "
                f"readings (Id {uid}). Choose another in the sidebar if needed."
            )

        # 2. features (Objective 2/4)
        stats = features.compute_stats(clean)
        clean_ma = features.rolling_features(clean)
        daily = features.daily_stats(clean)
        circ = features.circadian_profile(clean)

        # 3. sleep / activity context
        model = get_activity_model() if (use_activity_model and intensity_df is None) else None
        minute_hr = context.minute_hr_from_clean(canon)
        ctx = context.build_context(minute_hr, intensity_df, steps_df, sleep_df,
                                    user_id=uid, activity_model=model)
        if not np.isnan(ctx.resting_hr):
            stats.resting_hr = ctx.resting_hr
            stats.resting_method = (
                "25th percentile of HR during sustained (10+ min) still, awake minutes "
                f"(activity {ctx.intensity_source})"
            )
        dr = context.daily_resting(ctx)
        if not daily.empty and not dr.empty:
            daily = daily.drop(columns="resting_hr").merge(dr, on="date", how="left")
        if ctx.sleep_source == "observed" and not np.isnan(ctx.sleep_hr_low):
            stats.overnight_low_hr = ctx.sleep_hr_low

        peaks = features.detect_exertion_peaks(clean, resting_hr=stats.resting_hr)

        # 4. ML (Objective 3)
        feat = features.windowed_feature_matrix(clean, window=window)
        feat = context.add_window_context(feat, ctx, window)
        use_ctx = context_aware_anomalies and ctx.available and "residual_mean" in feat
        anom = ml.detect_anomalies(
            feat, method=anomaly_method, contamination=contamination,
            feature_cols=ml.CONTEXT_FEATURE_COLS if use_ctx else None,
        )
        clust = ml.cluster_patterns(feat, k=n_clusters)
        clf = ml.train_supervised(feat, label_col=label_col).to_dict() if label_col else None

        # 5. interpretation (Objective 4)
        interp = interpretation.interpret(
            stats, daily=daily, circadian=circ,
            anomaly_count=anom.n_anomalies, n_windows=anom.n_windows, age=age,
            ctx=ctx, model_meta=(model.meta if model is not None else None),
        )

        m_meta = {}
        am = get_activity_model()
        if am is not None:
            ev = am.meta.get("evaluation", {})
            m_meta = {
                "used": ctx.intensity_source == "estimated",
                "trained_on": am.meta.get("trained_on"),
                "algorithm": am.meta.get("algorithm"),
                "classes": am.meta.get("classes"),
                "evaluation": ev,
            }

        return AnalysisResult(
            clean=clean_ma, quality=qreport.to_dict(), stats=stats.to_dict(),
            daily=daily, circadian=circ, peaks=peaks,
            context=ctx.to_dict(), minutes=ctx.minutes,
            features=feat, anomalies=anom.to_dict(), clusters=clust.to_dict(),
            classifier=clf, activity_model=m_meta,
            interpretation=interp.to_dict(), ok=True,
        )
    except Exception as e:  # noqa: BLE001 - surface any error cleanly to the UI
        return AnalysisResult(ok=False, error=f"{type(e).__name__}: {e}")


def analyze_csv(path: str, **kwargs) -> AnalysisResult:
    """Convenience: load a CSV then analyze."""
    return analyze(preprocessing.load_csv(path), **kwargs)
