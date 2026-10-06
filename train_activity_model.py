"""Train + evaluate the activity model on the merged Fitbit minute table.
   python build_dataset.py <folders...>   (first)
   python train_activity_model.py"""
import json, time
import pandas as pd
from hr_backend import activity_model as am

t0 = time.time()
mt = pd.read_parquet("data/minute_table.parquet")
print("evaluating (leave-participants-out)...")
metrics = am.evaluate_lopo(mt)
print(json.dumps(metrics, indent=2))
print(f"eval {time.time()-t0:.0f}s; training final model on all participants...")
m = am.train_and_save(mt, metrics)
print("saved", am.MODEL_PATH, f"{am.MODEL_PATH.stat().st_size/1e6:.2f} MB", f"total {time.time()-t0:.0f}s")
