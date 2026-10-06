"""
hr_backend - analytical & ML backend for the smartwatch heart-rate app.

Frontend usage
--------------
    from hr_backend import analyze, analyze_csv

    result = analyze(uploaded_df)          # pandas DataFrame from st.file_uploader
    if result.ok:
        st.line_chart(result.clean.set_index("timestamp")["hr_smooth"])
        st.write(result.interpretation["summary_text"])
    else:
        st.error(result.error)

Modules
-------
    preprocessing  - cleaning, resampling, smoothing, quality (Objective 1)
    fitbit         - Fitbit file loaders, minute-level merge, upload detection
    context        - activity- and sleep-aware statistics
    activity_model - supervised activity-level model (trained, bundled in models/)
    features       - stats, rolling/daily/circadian, peaks (Objective 2/4)
    ml             - anomaly detection, clustering, classifier (Objective 3)
    interpretation - alerts + plain-text explanations (Objective 4)
    pipeline       - analyze() orchestrator (Objective 5)
    synthetic      - test-data generator (no download needed)
"""

from .pipeline import analyze, analyze_csv, AnalysisResult

__all__ = ["analyze", "analyze_csv", "AnalysisResult"]
__version__ = "0.2.0"
