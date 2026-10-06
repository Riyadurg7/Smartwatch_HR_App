"""
End-to-end self-test. Runs without the full dataset (uses bundled sample +
synthetic data).   python test_pipeline.py
"""
import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
from hr_backend import analyze, fitbit
from hr_backend.activity_model import ActivityModel
from hr_backend.evaluation import evaluate_anomaly_detection
from hr_backend.report import build_csv_bundle, build_pdf
from hr_backend.synthetic import generate

S = "sample_data/"
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 1. bundled activity model loads
m = ActivityModel.load()
check("activity model loads", m is not None,
      f"(LOPO accuracy {m.meta['evaluation']['accuracy']})" if m else "")

# 2. sample with observed activity + sleep
hr = pd.read_csv(S + "sample_heartrate.csv")
ctx = {"intensity_df": fitbit.load_intensity([S + "sample_minuteIntensitiesNarrow.csv"]),
       "steps_df": fitbit.load_steps([S + "sample_minuteStepsNarrow.csv"]),
       "sleep_df": fitbit.load_sleep([S + "sample_minuteSleep.csv"])}
r = analyze(hr, **ctx, age=30)
check("sample + context analyses", r.ok, r.error or "")
check("activity observed", r.context.get("intensity_source") == "observed")
check("sleep observed", r.context.get("sleep_source") == "observed")
check("context-aware anomalies", r.anomalies.get("context_aware"))
check("resting HR plausible", 40 < r.stats["resting_hr"] < 100, f"({r.stats['resting_hr']} bpm)")
check("disclaimer present", bool(r.interpretation.get("disclaimer")))

# 3. heart rate only -> model estimates activity
r2 = analyze(hr)
check("HR-only analyses", r2.ok, r2.error or "")
check("activity estimated by model", r2.context.get("intensity_source") == "estimated")
check("HR-only resting close to observed", abs(r2.stats["resting_hr"] - r.stats["resting_hr"]) < 6,
      f"({r2.stats['resting_hr']} vs {r.stats['resting_hr']})")

# 4. generic non-Fitbit CSV + synthetic
syn = generate(days=2)
gen = pd.DataFrame({"timestamp": pd.to_datetime(syn["Time"], format="%m/%d/%Y %I:%M:%S %p")
                    .dt.strftime("%Y-%m-%d %H:%M:%S"), "bpm": syn["Value"]})
r3 = analyze(gen)
check("generic CSV (timestamp,bpm)", r3.ok, r3.error or "")

# 5. reports
pdf = build_pdf(r, subject="self-test")
check("PDF report", pdf[:4] == b"%PDF", f"({len(pdf)//1024} KB)")
check("CSV bundle", len(build_csv_bundle(r)) >= 3)

# 6. benchmark both modes
from hr_backend import preprocessing
clean, _ = preprocessing.preprocess(hr)
e_plain = evaluate_anomaly_detection(clean, seed=0)
e_ctx = evaluate_anomaly_detection(clean, seed=0, context_frames={
    "intensity": ctx["intensity_df"], "steps": ctx["steps_df"], "sleep": ctx["sleep_df"]})
check("benchmark plain", e_plain.roc_auc == e_plain.roc_auc, f"(AUC {e_plain.roc_auc})")
check("benchmark context-aware", e_ctx.roc_auc == e_ctx.roc_auc, f"(AUC {e_ctx.roc_auc})")

# 7. JSON-serialisable payload for any frontend
json.dumps({"quality": r.quality, "stats": r.stats, "context": r.context,
            "interpretation": r.interpretation}, default=str)
check("JSON payload", True)

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
