"""Merge both Fitbit export periods into analysis-ready files in data/.
   python build_dataset.py "<Fitabase folder 1>" "<Fitabase folder 2>" """
import sys, time
import pandas as pd
from hr_backend import fitbit

folders = sys.argv[1:]
t0 = time.time()
d = fitbit.load_export_folders(folders)
print({k: len(v) for k, v in d.items()}, f"{time.time()-t0:.0f}s")

d["hr"].to_csv("data/heartrate_seconds_merged.csv", index=False)
mt = fitbit.build_minute_table(d["hr"], d["intensity"], d["steps"], d["sleep"])
mt.to_parquet("data/minute_table.parquet", index=False)
d["daily"].to_csv("data/dailyActivity_merged.csv", index=False)
print("minute table:", mt.shape, f"{time.time()-t0:.0f}s")

# participant profile
hr = d["hr"].copy()
hr["ts"] = pd.to_datetime(hr["Time"], format=fitbit.FITBIT_TS)
p = hr.groupby("Id").agg(rows=("Value", "size"), start=("ts", "min"), end=("ts", "max"))
p["days"] = ((p["end"] - p["start"]).dt.total_seconds() / 86400).round(1)
cov = mt.groupby("Id").agg(hr_minutes=("hr_mean", "size"),
                           intensity_cov=("intensity", lambda s: round(s.notna().mean() * 100, 1)),
                           sleep_minutes=("is_asleep", "sum"))
p = p.join(cov)
print(p[["rows", "days", "hr_minutes", "intensity_cov", "sleep_minutes"]].to_string())
print("participants with HR:", p.shape[0], " with sleep logged:", int((p["sleep_minutes"] > 0).sum()))
print("intensity label distribution (HR minutes):")
print(mt["intensity"].value_counts(dropna=False).sort_index())
