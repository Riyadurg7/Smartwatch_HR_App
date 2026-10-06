"""
Synthetic smartwatch heart-rate generator.

Used for development, testing and demos when no real dataset is downloaded.
Produces data in the same shape as Fitbit's heartrate_seconds_merged.csv
(columns: Id, Time, Value) so code paths are identical to real input.

The signal includes:
  - a circadian baseline (low at night, higher midday)
  - a nightly sleep dip
  - random daytime activity bursts (exercise bouts with recovery)
  - realistic sensor noise and occasional dropouts
  - optional injected anomalies (spikes/dropouts) for testing detection
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate(
    days: int = 3,
    sample_seconds: int = 15,
    resting_hr: float = 62.0,
    user_id: int = 1234567890,
    n_anomalies: int = 6,
    dropout_frac: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame with columns [Id, Time, Value] mimicking a Fitbit export."""
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2024-04-12 00:00:00")
    n = int(days * 24 * 3600 / sample_seconds)
    t = pd.date_range(start, periods=n, freq=f"{sample_seconds}s")

    hours = t.hour + t.minute / 60.0

    # Circadian baseline: ~ -8 bpm at ~4am, +12 bpm mid-afternoon.
    circadian = 10 * np.sin((hours - 10) / 24 * 2 * np.pi)
    # Deeper sleep dip between 00:00 and 06:00.
    sleep = np.where((hours >= 0) & (hours < 6), -8.0, 0.0)

    base = resting_hr + circadian + sleep

    # Random daytime exercise bouts (elevated HR with exponential recovery).
    activity = np.zeros(n)
    day_index = (t - start).days
    for d in range(days):
        n_bouts = rng.integers(1, 4)
        for _ in range(n_bouts):
            # start sometime between 7am and 9pm
            bout_hour = rng.uniform(7, 21)
            center = int((d * 24 + bout_hour) * 3600 / sample_seconds)
            if center >= n:
                continue
            amp = rng.uniform(30, 70)          # bpm above baseline
            rise = int(rng.uniform(3, 8) * 60 / sample_seconds)
            length = int(rng.uniform(15, 45) * 60 / sample_seconds)
            for k in range(length):
                idx = center + k
                if idx >= n:
                    break
                if k < rise:
                    frac = k / max(1, rise)
                else:
                    frac = np.exp(-(k - rise) / (length / 2))
                activity[idx] += amp * frac

    signal = base + activity
    # Physiological noise
    signal += rng.normal(0, 1.8, n)
    signal = np.clip(signal, 35, 210)

    df = pd.DataFrame({"Id": user_id, "Time": t, "Value": np.round(signal, 0)})

    # Inject anomalies (sudden implausible spikes/drops) for detection testing.
    if n_anomalies > 0:
        pos = rng.choice(n, size=n_anomalies, replace=False)
        for p in pos:
            df.loc[p, "Value"] = rng.choice([rng.uniform(180, 235), rng.uniform(30, 42)])

    # Simulate sensor dropouts (missing rows).
    if dropout_frac > 0:
        keep = rng.random(n) > dropout_frac
        df = df[keep].reset_index(drop=True)

    # Fitbit stores Time as "M/D/YYYY h:mm:ss AM/PM"
    df["Time"] = df["Time"].dt.strftime("%-m/%-d/%Y %-I:%M:%S %p")
    return df


if __name__ == "__main__":
    out = generate()
    out.to_csv("synthetic_heartrate.csv", index=False)
    print(f"Wrote synthetic_heartrate.csv  ({len(out)} rows)")
    print(out.head())
