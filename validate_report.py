"""
Reproducible validation on the full FitBit Fitness Tracker Data (both periods).

    python validate_report.py "<Fitabase Data 3.12.16-4.11.16>" "<Fitabase Data 4.12.16-5.12.16>"

Prints markdown tables used in VALIDATION.md and writes validation_results.json.
Sections:
  1. per-participant analysis with observed activity + sleep context
  2. activity model (leave-participants-out metrics, from training)
  3. HR-only path: resting HR / active time estimated by a model that never saw
     that participant, compared with the values from the real activity files
  4. anomaly detector: injected-anomaly benchmark (plain vs context-aware) and
     how many flags are "just exercise"
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from hr_backend import analyze, fitbit, preprocessing, context, ml, features
from hr_backend import activity_model as am
from hr_backend.evaluation import evaluate_anomaly_detection

T0 = time.time()
folders = sys.argv[1:]
d = fitbit.load_export_folders(folders)
hr_all = d["hr"]
users = sorted(hr_all["Id"].unique())
ctxf = {"intensity": d["intensity"], "steps": d["steps"], "sleep": d["sleep"]}
out = {}


def log(msg):
    print(f"[{time.time()-T0:5.0f}s] {msg}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------ 1
rows = []
for uid in users:
    r = analyze(hr_all[hr_all.Id == uid], **{f"{k}_df": v for k, v in ctxf.items()})
    if not r.ok:
        rows.append(dict(user=uid, error=r.error)); continue
    q, s, c, a = r.quality, r.stats, r.context, r.anomalies
    act = {x["activity"]: x for x in c["hr_by_activity"]}
    rows.append(dict(
        user=str(uid), days=round(q["duration_hours"] / 24, 1), quality=q["quality_score"],
        coverage=q["coverage_pct"], resting=s["resting_hr"],
        sleep_hr=c["sleep_hr_mean"], sleep_src="obs" if c["sleep_source"] == "observed" else "assumed",
        sed_hr=act.get("sedentary", {}).get("median_hr"), active_hr=act.get("active", {}).get("median_hr"),
        pct_active=act.get("active", {}).get("pct"),
        rest_episodes=len(c["elevated_rest_episodes"]),
        anomalies=f"{a['n_anomalies']}/{a['n_windows']}"))
t1 = pd.DataFrame(rows)
out["participants"] = t1.to_dict(orient="records")
print("## 1. Participants (observed activity + sleep context)\n")
print(t1.to_markdown(index=False))
log("section 1 done")

# ------------------------------------------------------------------ 2
meta = am.ActivityModel.load().meta
out["activity_model"] = meta["evaluation"]
ev = meta["evaluation"]
print("\n## 2. Activity model (leave-participants-out)\n")
print(pd.DataFrame([
    dict(model="Always 'sedentary'", accuracy=ev["baseline_always_sedentary_accuracy"], balanced_accuracy=0.333, f1_macro=None),
    dict(model="HR threshold (logistic on relative HR)", **ev["baseline_hr_threshold"]),
    dict(model="Trained model (gradient boosting)", accuracy=ev["accuracy"],
         balanced_accuracy=ev["balanced_accuracy"], f1_macro=ev["f1_macro"]),
]).to_markdown(index=False))
print(f"\nROC-AUC macro {ev['roc_auc_macro_ovr']}, active-vs-rest {ev['roc_auc_active_vs_rest']}")
print(pd.DataFrame(ev["per_class"]).T.to_markdown())

# ------------------------------------------------------------------ 3
mt = pd.read_parquet("data/minute_table.parquet").dropna(subset=["intensity", "hr_mean"]).reset_index(drop=True)
X = am.make_features(mt)
y = am.labels_from_intensity(mt["intensity"]).astype(int).values
rows = []
for uid in users:
    tr = (mt["Id"] != uid).values
    model = am._new_model().fit(X[tr], y[tr])
    held = am.ActivityModel(model, {})
    canon = preprocessing.standardize(hr_all[hr_all.Id == uid])
    mhr = context.minute_hr_from_clean(canon)
    obs = context.build_context(mhr, d["intensity"], d["steps"], d["sleep"], user_id=uid)
    est = context.build_context(mhr, None, None, d["sleep"], user_id=uid, activity_model=held)
    pa = lambda c: {x["activity"]: x["pct"] for x in c.hr_by_activity}.get("active")
    rows.append(dict(user=str(uid), resting_observed=obs.resting_hr, resting_estimated=est.resting_hr,
                     diff=round(est.resting_hr - obs.resting_hr, 1) if not np.isnan(obs.resting_hr) and not np.isnan(est.resting_hr) else None,
                     active_pct_observed=pa(obs), active_pct_estimated=pa(est)))
t3 = pd.DataFrame(rows)
mae = t3["diff"].abs().mean()
out["hr_only_path"] = dict(table=t3.to_dict(orient="records"), resting_mae=round(float(mae), 2))
print("\n## 3. HR-only path vs real activity files (model never saw the participant)\n")
print(t3.to_markdown(index=False))
print(f"\nMean absolute difference in resting HR: {mae:.1f} bpm")
log("section 3 done")

# ------------------------------------------------------------------ 4
big = sorted(users, key=lambda u: -(hr_all.Id == u).sum())[:5]
bench = []
for method in ["isolation_forest", "lof"]:
    for mode in ["plain", "context"]:
        agg = []
        for uid in big:
            clean, _ = preprocessing.preprocess(hr_all[hr_all.Id == uid])
            es = [evaluate_anomaly_detection(clean, method=method, contamination=0.05, seed=sd,
                                             context_frames=ctxf if mode == "context" else None, user_id=uid)
                  for sd in range(3)]
            agg.append([np.nanmean([getattr(e, m) for e in es]) for m in ["precision", "recall", "f1", "roc_auc", "pr_auc"]])
        agg = np.array(agg).mean(0)
        bench.append(dict(method=method, features=mode, precision=round(agg[0], 3), recall=round(agg[1], 3),
                          f1=round(agg[2], 3), roc_auc=round(agg[3], 3), pr_auc=round(agg[4], 3)))
        log(f"bench {method} {mode}")
t4 = pd.DataFrame(bench)
out["anomaly_benchmark"] = t4.to_dict(orient="records")
print("\n## 4a. Injected-anomaly benchmark (5 largest participants x 3 seeds)\n")
print(t4.to_markdown(index=False))

# share of flagged windows that are mostly exercise (real data, no injection)
rows = []
for uid in users:
    canon = preprocessing.standardize(hr_all[hr_all.Id == uid])
    clean, _ = preprocessing.preprocess(canon, time_col="timestamp", hr_col="hr")
    c = context.build_context(context.minute_hr_from_clean(canon), d["intensity"], d["steps"], d["sleep"], user_id=uid)
    feat = context.add_window_context(features.windowed_feature_matrix(clean), c, "10min")
    if feat.empty or "mean_activity" not in feat:
        continue
    plain = ml.detect_anomalies(feat).table
    ctxa = ml.detect_anomalies(feat, feature_cols=ml.CONTEXT_FEATURE_COLS).table
    ex = feat["mean_activity"] >= 1.0  # window spent mostly in light-to-active movement
    still = (feat["mean_activity"] < 0.2) & (feat["asleep_frac"] < 0.5)
    rows.append(dict(user=str(uid),
                     plain_flags_during_movement=round(100 * (plain.is_anomaly & ex).sum() / max(1, plain.is_anomaly.sum()), 1),
                     context_flags_during_movement=round(100 * (ctxa.is_anomaly & ex).sum() / max(1, ctxa.is_anomaly.sum()), 1),
                     plain_flags_while_still=round(100 * (plain.is_anomaly & still).sum() / max(1, plain.is_anomaly.sum()), 1),
                     context_flags_while_still=round(100 * (ctxa.is_anomaly & still).sum() / max(1, ctxa.is_anomaly.sum()), 1)))
t5 = pd.DataFrame(rows)
out["flag_composition"] = dict(table=t5.to_dict(orient="records"), mean=t5.drop(columns="user").mean().round(1).to_dict())
print("\n## 4b. What gets flagged on real data (% of flagged windows)\n")
print(t5.to_markdown(index=False))
print("\nMean:", t5.drop(columns="user").mean().round(1).to_dict())

json.dump(out, open("validation_results.json", "w"), indent=2, default=str)
log("all done")
