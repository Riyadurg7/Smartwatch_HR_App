"""
Smartwatch Heart-Rate Analyzer - deployable Streamlit application.

Implements the Swastome project objectives end to end:
  1 preprocessing & quality      2 visualization dashboard
  3 ML: anomaly detection, clustering, supervised activity model
  4 automated interpretation     5 configurable UI + downloadable reports
  6 reproducible deployment (Streamlit Community Cloud / Hugging Face Spaces)

Run locally:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hr_backend import analyze, fitbit
from hr_backend.evaluation import evaluate_anomaly_detection
from hr_backend.pipeline import get_activity_model
from hr_backend.report import build_csv_bundle, build_pdf
from hr_backend.synthetic import generate

st.set_page_config(page_title="Smartwatch HR Analyzer", page_icon="❤️", layout="wide")

SAMPLE_DIR = Path(__file__).parent / "sample_data"
ACT_COLORS = {"sedentary": "#94a3b8", "light": "#38bdf8", "active": "#f97316", "asleep": "#6366f1"}

st.title("❤️ Smartwatch Heart-Rate Analyzer")
st.caption(
    "Upload a smartwatch heart-rate export for preprocessing, visualization, activity- and "
    "sleep-aware analysis, anomaly detection and a plain-language report. "
    "Informational only - not a medical device."
)


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_sample(with_context: bool):
    hr = pd.read_csv(SAMPLE_DIR / "sample_heartrate.csv")
    if not with_context:
        return hr, {}
    return hr, {
        "intensity": fitbit.load_intensity([SAMPLE_DIR / "sample_minuteIntensitiesNarrow.csv"]),
        "steps": fitbit.load_steps([SAMPLE_DIR / "sample_minuteStepsNarrow.csv"]),
        "sleep": fitbit.load_sleep([SAMPLE_DIR / "sample_minuteSleep.csv"]),
    }


@st.cache_data(show_spinner="Analyzing...")
def run_analysis(hr, ctx, user_id, window, method, contamination, k_choice, age, ctx_anom):
    return analyze(
        hr, user_id=user_id,
        intensity_df=ctx.get("intensity"), steps_df=ctx.get("steps"), sleep_df=ctx.get("sleep"),
        window=window, anomaly_method=method, contamination=contamination,
        n_clusters=None if k_choice == "auto" else int(k_choice),
        age=int(age) if age else None, context_aware_anomalies=ctx_anom,
    )


# --------------------------------------------------------------------------- #
# sidebar (Objective 5: configurable parameters)
# --------------------------------------------------------------------------- #
hr_df, ctx_frames, load_err = None, {}, None
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", [
        "Sample: 7 days with activity & sleep",
        "Sample: heart rate only",
        "Upload my files",
        "Synthetic (generated)",
    ], help="Samples are one real, anonymised participant from the FitBit Fitness Tracker Data. "
            "'Heart rate only' shows how the trained model estimates activity when no activity file exists.")

    if source.startswith("Sample: 7"):
        hr_df, ctx_frames = load_sample(True)
    elif source.startswith("Sample: heart"):
        hr_df, ctx_frames = load_sample(False)
    elif source.startswith("Synthetic"):
        hr_df = generate(days=4)
    else:
        up = st.file_uploader("Heart-rate CSV (required)", type="csv",
                              help="Fitbit heartrate_seconds_merged.csv, or any CSV with a timestamp and a BPM column.")
        extras = st.file_uploader(
            "Optional: activity / steps / sleep CSVs", type="csv", accept_multiple_files=True,
            help="Fitbit minuteIntensitiesNarrow, minuteStepsNarrow and minuteSleep files. "
                 "Detected automatically from their columns.")
        if up is not None:
            try:
                hr_df = pd.read_csv(up)
            except Exception as e:
                load_err = f"Could not read heart-rate CSV: {e}"
        if extras:
            comp = fitbit.read_companions([pd.read_csv(f) for f in extras])
            ctx_frames = {k: v for k, v in comp.items() if k != "unknown" and v is not None}
            st.caption("Detected: " + (", ".join(ctx_frames) or "nothing usable")
                       + (f" ({comp['unknown']} file(s) not recognised)" if comp["unknown"] else ""))

    user_id = None
    if hr_df is not None and "Id" in hr_df.columns and hr_df["Id"].nunique() > 1:
        counts = hr_df["Id"].value_counts()
        user_id = st.selectbox("Participant", counts.index.tolist(),
                               format_func=lambda i: f"{i} ({counts[i]:,} readings)")

    st.header("2. Settings")
    window = st.selectbox("Analysis window", ["5min", "10min", "30min"], index=1)
    method = st.selectbox("Anomaly method", ["isolation_forest", "lof"],
                          format_func={"isolation_forest": "Isolation Forest", "lof": "Local Outlier Factor"}.get)
    contamination = st.slider("Expected anomaly fraction", 0.01, 0.20, 0.05, 0.01)
    ctx_anom = st.checkbox("Activity-aware anomaly detection", value=True,
                           help="Judge heart rate relative to what is normal for the activity level, "
                                "so workouts are not flagged just for being high.")
    k_choice = st.selectbox("Clusters", ["auto", 2, 3, 4, 5])
    age = st.number_input("Age (optional, for max-HR %)", 0, 120, 0)
    run = st.button("Run analysis", type="primary", width="stretch")

if run:
    if load_err:
        st.error(load_err); st.stop()
    if hr_df is None:
        st.info("Upload a heart-rate CSV in the sidebar first."); st.stop()
    st.session_state["res"] = run_analysis(hr_df, ctx_frames, user_id, window, method,
                                           contamination, k_choice, age, ctx_anom)
    st.session_state["ctx_frames"] = ctx_frames
    st.session_state["uid"] = user_id
    st.session_state["src"] = source

if "res" not in st.session_state:
    st.info("Choose a data source and press **Run analysis** in the sidebar. "
            "Try the 7-day sample first to see every feature.")
    st.stop()

result = st.session_state["res"]
if not result.ok:
    st.error(f"Analysis failed: {result.error}"); st.stop()

clean, q, s, c = result.clean, result.quality, result.stats, result.context
anom = pd.DataFrame(result.anomalies.get("table", []))
minutes = result.minutes

tabs = st.tabs(["Overview", "Activity & Sleep", "Visualizations", "Machine Learning",
                "Model & Evaluation", "Report & Export", "About"])

# ================================ OVERVIEW ================================= #
with tabs[0]:
    if q.get("user_id"):
        st.caption(f"Participant {q['user_id']}  |  {q['start'][:16]} to {q['end'][:16]}")
    st.subheader("Data quality")
    cols = st.columns(4)
    cols[0].metric("Quality", f"{q['quality_score']}/100", q["quality_label"])
    cols[1].metric("Coverage", f"{q['coverage_pct']}%")
    cols[2].metric("Duration", f"{q['duration_hours']/24:.1f} days")
    cols[3].metric("Sampling grid", q["resample_rule"])
    for w in q["warnings"]:
        st.warning(w, icon="⚠️")

    st.subheader("Key numbers")
    cols = st.columns(4)
    cols[0].metric("Resting HR", f"{s['resting_hr']} bpm", help=s.get("resting_method"))
    cols[1].metric("Average HR", f"{s['mean_hr']} bpm")
    cols[2].metric("Peak HR (99th pct)", f"{s['max_hr']} bpm")
    cols[3].metric("Lowest during sleep", f"{s.get('overnight_low_hr', 'n/a')} bpm")

    st.subheader("What this means")
    it = result.interpretation
    st.write(it["summary_text"])
    if it["trend_text"]:
        st.write(it["trend_text"])
    for al in it["alerts"]:
        (st.warning if al["level"] == "attention" else st.info)(al["message"])
    st.caption(it["disclaimer"])

# ============================ ACTIVITY & SLEEP ============================= #
with tabs[1]:
    if not c.get("available"):
        st.info("No activity context available for this recording.")
    else:
        src = {"observed": "from your activity file", "estimated": "estimated by the trained model"}
        st.caption(f"Activity: {src.get(c['intensity_source'], c['intensity_source'])}  |  "
                   f"Sleep: {c['sleep_source']}")
        by = pd.DataFrame(c["hr_by_activity"])
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Heart rate by activity")
            st.plotly_chart(px.bar(by, x="activity", y="median_hr", color="activity",
                                   color_discrete_map=ACT_COLORS, text="median_hr",
                                   labels={"median_hr": "median bpm", "activity": ""})
                            .update_layout(height=320, showlegend=False, margin=dict(t=10)), width="stretch")
        with c2:
            st.subheader("Where the time went")
            st.plotly_chart(px.pie(by, names="activity", values="minutes", color="activity",
                                   color_discrete_map=ACT_COLORS, hole=0.5)
                            .update_layout(height=320, margin=dict(t=10)), width="stretch")

        st.subheader("Heart rate coloured by activity")
        if not minutes.empty:
            mm = minutes.copy()
            names = {0: "sedentary", 1: "light", 2: "active"}
            mm["state"] = np.where(mm["is_asleep"], "asleep", mm["activity"].map(names))
            fig = px.scatter(mm, x="minute", y="hr_mean", color="state", color_discrete_map=ACT_COLORS,
                             render_mode="webgl", labels={"hr_mean": "bpm", "minute": ""})
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(height=380, margin=dict(t=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, width="stretch")

        st.subheader("High heart rate while still")
        eps = c.get("elevated_rest_episodes", [])
        if eps:
            st.write(f"Periods of 10+ minutes where heart rate stayed above "
                     f"{max(100, (s['resting_hr'] or 0) + 30):.0f} bpm while sitting still and awake:")
            st.dataframe(pd.DataFrame(eps), width="stretch", hide_index=True)
        else:
            st.success("No sustained periods of high heart rate while still. ")

# ============================ VISUALIZATIONS =============================== #
with tabs[2]:
    st.subheader("Heart-rate timeline")
    show_ma = st.checkbox("Show moving averages", value=True)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=clean["timestamp"], y=clean["hr_smooth"], mode="lines",
                               name="HR (smoothed)", line=dict(width=1)))
    if show_ma:
        for col, nm in [("ma_30min", "30-min avg"), ("ma_1h", "1-hour avg")]:
            if col in clean:
                fig.add_trace(go.Scattergl(x=clean["timestamp"], y=clean[col], mode="lines",
                                           name=nm, line=dict(width=1.5)))
    if not anom.empty and "is_anomaly" in anom:
        fl = anom[anom["is_anomaly"]]
        fig.add_trace(go.Scatter(x=fl["window_start"], y=fl["mean"], mode="markers", name="unusual window",
                                 marker=dict(color="red", size=8, symbol="x")))
    fig.update_layout(height=380, margin=dict(t=10), yaxis_title="bpm", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribution")
        st.plotly_chart(px.histogram(clean, x="hr_smooth", nbins=40, labels={"hr_smooth": "bpm"})
                        .update_layout(height=300, margin=dict(t=10)), width="stretch")
    with c2:
        st.subheader("Average by hour of day")
        st.plotly_chart(px.line(result.circadian, x="hour", y="mean_hr", markers=True,
                                labels={"mean_hr": "avg bpm", "hour": "hour of day"})
                        .update_layout(height=300, margin=dict(t=10)), width="stretch")

    st.subheader("Daily trend")
    if not result.daily.empty:
        st.plotly_chart(px.line(result.daily, x="date", y=["resting_hr", "mean_hr", "max_hr"], markers=True)
                        .update_layout(height=300, margin=dict(t=10), yaxis_title="bpm"), width="stretch")

    st.subheader("Exercise episodes and recovery")
    if not result.peaks.empty:
        st.dataframe(result.peaks, width="stretch", hide_index=True)
        st.caption("Recovery = bpm drop in the 1-2 minutes after the peak. A bigger drop usually means better fitness.")
    else:
        st.write("No sustained exertion episodes detected.")

# =========================== MACHINE LEARNING ============================== #
with tabs[3]:
    a = result.anomalies
    st.subheader("Unusual-pattern detection")
    st.write(
        f"**{'Isolation Forest' if a['method']=='isolation_forest' else 'Local Outlier Factor'}** "
        f"split the recording into {window} windows and flagged **{a['n_anomalies']} of {a['n_windows']}** "
        "as unusual. " + ("Each window was judged against what is normal *for its activity level and "
                          "sleep*, so workouts are not flagged just for being high."
                          if a.get("context_aware") else "Windows were judged on heart-rate shape only.")
    )
    if not anom.empty:
        st.plotly_chart(px.histogram(anom, x="anomaly_score", color="is_anomaly", nbins=40,
                                     labels={"anomaly_score": "how unusual (higher = more)"})
                        .update_layout(height=260, margin=dict(t=10)), width="stretch")
        show = ["window_start", "mean", "min", "max"] + \
               [x for x in ["mean_activity", "expected_hr", "residual_mean"] if x in anom] + ["anomaly_score"]
        st.dataframe(anom[anom["is_anomaly"]][show].sort_values("anomaly_score", ascending=False).head(50)
                     .round(2), width="stretch", hide_index=True)

    st.subheader("Behaviour groups (clustering)")
    cl = result.clusters
    st.write(f"KMeans grouped the windows into **{cl['k']}** patterns, numbered from lowest to highest heart rate:")
    st.dataframe(pd.DataFrame(cl["profile"]), width="stretch", hide_index=True)
    ct = pd.DataFrame(cl["table"])
    if not ct.empty and "cluster" in ct:
        st.plotly_chart(px.scatter(ct, x="mean", y="std", color=ct["cluster"].astype(str),
                                   labels={"mean": "window mean HR", "std": "window variability", "color": "group"})
                        .update_layout(height=320, margin=dict(t=10)), width="stretch")

# ========================== MODEL & EVALUATION ============================= #
with tabs[4]:
    st.subheader("Supervised activity model")
    am = result.activity_model
    if not am:
        st.warning("Activity model file not found - heart-rate-only uploads will skip activity estimation.")
    else:
        ev = am["evaluation"]
        st.write(
            "Predicts each minute's activity level (**sedentary / light / active**) from heart rate alone. "
            f"Trained on {am['trained_on']} using real Fitbit activity labels. Scores below come from "
            "**leave-participants-out** testing: the model is always tested on people it never saw."
        )
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{ev['accuracy']:.0%}",
                       f"+{(ev['accuracy']-ev['baseline_always_sedentary_accuracy'])*100:.0f} pts vs guessing")
        cols[1].metric("Balanced accuracy", f"{ev['balanced_accuracy']:.0%}")
        cols[2].metric("Macro F1", f"{ev['f1_macro']:.2f}")
        cols[3].metric("ROC-AUC (active vs rest)", f"{ev['roc_auc_active_vs_rest']:.2f}")
        base = pd.DataFrame([
            {"model": "Always guess 'sedentary'", "accuracy": ev["baseline_always_sedentary_accuracy"],
             "balanced accuracy": 0.333, "macro F1": None},
            {"model": "Heart-rate threshold rule", "accuracy": ev["baseline_hr_threshold"]["accuracy"],
             "balanced accuracy": ev["baseline_hr_threshold"]["balanced_accuracy"],
             "macro F1": ev["baseline_hr_threshold"]["f1_macro"]},
            {"model": "This model", "accuracy": ev["accuracy"], "balanced accuracy": ev["balanced_accuracy"],
             "macro F1": ev["f1_macro"]},
        ])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Compared with simple baselines**")
            st.dataframe(base, width="stretch", hide_index=True)
            st.markdown("**Per class**")
            st.dataframe(pd.DataFrame(ev["per_class"]).T, width="stretch")
        with c2:
            cm = np.array(ev["confusion_matrix"])
            cmn = cm / cm.sum(1, keepdims=True)
            st.markdown("**Confusion matrix** (rows = truth, % of each row)")
            st.plotly_chart(px.imshow(cmn, x=am["classes"], y=am["classes"], text_auto=".0%",
                                      color_continuous_scale="Blues", labels=dict(x="predicted", y="actual"))
                            .update_layout(height=330, margin=dict(t=10), coloraxis_showscale=False),
                            width="stretch")
        if am.get("used"):
            st.info("This analysis used the model's estimates because no activity file was provided.")

    st.subheader("Anomaly-detector benchmark")
    st.write(
        "The dataset has no doctor-confirmed anomaly labels, so we hide known fake problems (spikes, "
        "dropouts, long stretches of raised heart rate) in this real recording and check how many the "
        "detector finds. Treat it as a quality check on the method, not a medical validation."
    )
    if st.button("Run benchmark on this recording"):
        with st.spinner("Hiding anomalies and scoring (about 30 s)..."):
            cf = st.session_state.get("ctx_frames") or None
            model = get_activity_model() if (not cf or "intensity" not in cf) else None
            ctx_on = result.anomalies.get("context_aware", False)
            evals = [evaluate_anomaly_detection(
                clean[["timestamp", "hr_resampled", "hr_smooth", "is_filled"]], method=result.anomalies["method"],
                contamination=result.anomalies["contamination"], window=window, seed=sd,
                context_frames=cf if ctx_on else None, user_id=q.get("user_id"),
                activity_model=model if ctx_on else None) for sd in range(3)]
        m = lambda k: float(np.nanmean([getattr(e, k) for e in evals]))
        cols = st.columns(5)
        cols[0].metric("Recall", f"{m('recall'):.2f}", help="Share of hidden problems that were found")
        cols[1].metric("Precision", f"{m('precision'):.2f}", help="Share of flags that were hidden problems")
        cols[2].metric("F1", f"{m('f1'):.2f}")
        cols[3].metric("ROC-AUC", f"{m('roc_auc'):.2f}", help="0.5 = coin flip, 1.0 = perfect ranking")
        cols[4].metric("PR-AUC", f"{m('pr_auc'):.2f}")
        st.caption("Precision is limited by design: the detector flags a fixed share of windows, and real "
                   "recordings contain genuine unusual moments that were not on the hidden list.")

# ============================ REPORT & EXPORT ============================== #
with tabs[5]:
    st.subheader("Downloadable report")
    try:
        pdf = build_pdf(result, subject=f"Participant {q.get('user_id') or ''}".strip())
        st.download_button("⬇️ Download PDF report", pdf, "heart_rate_report.pdf", "application/pdf", type="primary")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
    st.markdown("**CSV exports**")
    for name, data in build_csv_bundle(result).items():
        st.download_button(f"⬇️ {name}", data, name, "text/csv", key=name)

# ================================= ABOUT =================================== #
with tabs[6]:
    st.subheader("How it maps to the project objectives")
    st.markdown("""
| Objective | Where in the app |
|---|---|
| 1 Preprocessing | cleaning, resampling, smoothing, quality score (Overview) |
| 2 Visualization | timeline, histogram, moving averages, hour-of-day, daily trend, recovery (Visualizations, Activity & Sleep) |
| 3 Machine learning | Isolation Forest / LOF anomalies, KMeans clustering, supervised activity model with accuracy / precision / recall / F1 / ROC-AUC (Machine Learning, Model & Evaluation) |
| 4 Interpretation | resting HR, sleep HR, peak, variability, alerts, plain-language summary (Overview) |
| 5 Configurable UI + reports | sidebar settings, PDF + CSV export |
| 6 Deployment | Streamlit Community Cloud / Hugging Face Spaces |
""")
    st.subheader("Limitations")
    st.markdown("""
- **Not a medical device** and not a diagnosis.
- **Variability (HRV) is an approximation** from BPM; true HRV needs beat-to-beat data.
- **Without a sleep file**, sleep is assumed to be 00:00-06:00.
- **Activity estimates** from heart rate alone are about 81% accurate on new people; light activity is the hardest class.
- **Anomaly accuracy** is measured with hidden fake anomalies, because the public data has no clinical labels.
- Training data: 15 adults, 2016, one device type - results may differ for other watches and populations.
""")
    st.caption(result.interpretation["disclaimer"])
