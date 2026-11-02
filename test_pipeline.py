"""
End-to-end self-test. Needs no dataset download: uses the bundled sample_data/
(skipped with a warning if missing) and synthetic data.

    python test_pipeline.py
"""
import json
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
from hr_backend import analyze, fitbit, preprocessing
from hr_backend.activity_model import ActivityModel
from hr_backend.evaluation import evaluate_anomaly_detection
from hr_backend.report import build_csv_bundle, build_pdf
from hr_backend.synthetic import generate

S = Path(__file__).parent / "sample_data"
SAMPLE_FILES = ["sample_heartrate.csv", "sample_minuteIntensitiesNarrow.csv",
                "sample_minuteStepsNarrow.csv", "sample_minuteSleep.csv"]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# 1. synthetic data works on every OS (Fitbit-style timestamps, no "%-m")
syn = generate(days=2)
check("synthetic data generates", len(syn) > 1000, f"({len(syn)} rows, e.g. '{syn['Time'].iloc[0]}')")
check("synthetic timestamps parse as Fitbit format",
      pd.to_datetime(syn["Time"], format=fitbit.FITBIT_TS, errors="coerce").notna().all())
check("synthetic analyses", analyze(syn).ok)

# 2. bundled activity model loads (or explains why not)
m = ActivityModel.load()
check("activity model loads", m is not None,
      f"(LOPO accuracy {m.meta['evaluation']['accuracy']})" if m else f"({ActivityModel.load_problem})")

# 3. bundled sample with observed activity + sleep
missing = [f for f in SAMPLE_FILES if not (S / f).exists()]
if missing:
    print(f"[SKIP] sample checks - missing {missing} in {S}. Commit the sample_data/ folder.")
    hr = syn
    r = analyze(hr, age=30)
    check("fallback analysis on synthetic data", r.ok, r.error or "")
else:
    hr = pd.read_csv(S / "sample_heartrate.csv")
    ctx = {"intensity_df": fitbit.load_intensity([S / "sample_minuteIntensitiesNarrow.csv"]),
           "steps_df": fitbit.load_steps([S / "sample_minuteStepsNarrow.csv"]),
           "sleep_df": fitbit.load_sleep([S / "sample_minuteSleep.csv"])}
    r = analyze(hr, **ctx, age=30)
    check("sample + context analyses", r.ok, r.error or "")
    check("activity observed", r.context.get("intensity_source") == "observed")
    check("sleep observed", r.context.get("sleep_source") == "observed")
    check("context-aware anomalies", r.anomalies.get("context_aware"))
    check("resting HR plausible", 40 < r.stats["resting_hr"] < 100, f"({r.stats['resting_hr']} bpm)")

    # heart rate only -> model estimates activity
    r2 = analyze(hr)
    check("HR-only analyses", r2.ok, r2.error or "")
    if m is not None:
        check("activity estimated by model", r2.context.get("intensity_source") == "estimated")
        check("HR-only resting close to observed", abs(r2.stats["resting_hr"] - r.stats["resting_hr"]) < 6,
              f"({r2.stats['resting_hr']} vs {r.stats['resting_hr']})")

    clean, _ = preprocessing.preprocess(hr)
    e_ctx = evaluate_anomaly_detection(clean, seed=0, context_frames={
        "intensity": ctx["intensity_df"], "steps": ctx["steps_df"], "sleep": ctx["sleep_df"]})
    check("benchmark context-aware", e_ctx.roc_auc == e_ctx.roc_auc, f"(AUC {e_ctx.roc_auc})")

check("disclaimer present", bool(r.interpretation.get("disclaimer")))

# 4. app keeps working when the model cannot be used (e.g. other scikit-learn version)
import hr_backend.pipeline as pl
saved = pl._MODEL_CACHE.copy()
pl._MODEL_CACHE["m"] = None
ActivityModel.load_problem = "simulated version mismatch"
r_nomodel = analyze(hr)
check("works without the model", r_nomodel.ok and r_nomodel.context.get("intensity_source") == "none",
      f"(activity_model={r_nomodel.activity_model})")
pl._MODEL_CACHE.clear(); pl._MODEL_CACHE.update(saved)

# 5. generic non-Fitbit CSV
gen = pd.DataFrame({"timestamp": pd.to_datetime(syn["Time"], format=fitbit.FITBIT_TS)
                    .dt.strftime("%Y-%m-%d %H:%M:%S"), "bpm": syn["Value"]})
r3 = analyze(gen)
check("generic CSV (timestamp,bpm)", r3.ok, r3.error or "")

# 6. reports
pdf = build_pdf(r, subject="self-test")
check("PDF report", pdf[:4] == b"%PDF", f"({len(pdf)//1024} KB)")
check("CSV bundle", len(build_csv_bundle(r)) >= 3)
clean_p, _ = preprocessing.preprocess(hr)
e_plain = evaluate_anomaly_detection(clean_p, seed=0)
check("benchmark plain", e_plain.roc_auc == e_plain.roc_auc, f"(AUC {e_plain.roc_auc})")

# 7. JSON-serialisable payload for any frontend
json.dumps({"quality": r.quality, "stats": r.stats, "context": r.context,
            "interpretation": r.interpretation}, default=str)
check("JSON payload", True)

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
