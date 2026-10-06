"""
Fallback download: heart-rate file only, ONE export period (Mar-Apr 2016).

Fetches the public OpenML mirror (CC0) of the Kaggle 'FitBit Fitness Tracker
Data' heartrate_seconds_merged.csv into data/. For the full dataset (both
periods, plus activity, steps and sleep, which the model training and
validation need) download the Kaggle zip and run build_dataset.py instead.

    python download_data.py
"""
import re
import urllib.request
from pathlib import Path

URL = "https://openml.org/data/v1/download/22120547/FitBit_HeartRate.arff"
OUT = Path("data/heartrate_seconds_merged_openml.csv")


def main():
    OUT.parent.mkdir(exist_ok=True)
    print("Downloading (~41 MB) ...")
    raw = urllib.request.urlopen(URL, timeout=120).read().decode("utf-8", "replace")

    rows, in_data = [], False
    for line in raw.splitlines():
        if not in_data:
            if line.strip().lower() == "@data":
                in_data = True
            continue
        m = re.match(r"^([0-9]+),'([^']+)',([0-9]+)$", line.strip())
        if m:
            rows.append(f"{m.group(1)},{m.group(2)},{m.group(3)}")

    with open(OUT, "w") as f:
        f.write("Id,Time,Value\n")
        f.write("\n".join(rows))
    print(f"Wrote {OUT}  ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
