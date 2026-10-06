# Smartwatch Heart-Rate Analyzer

A web application that takes an exported smartwatch heart-rate file and returns
cleaned data, charts, activity- and sleep-aware statistics, unusual-pattern
detection, a trained activity model, and a plain-language PDF report. Built to
the Swastome project specification (Objectives 1-6) on the **FitBit Fitness
Tracker Data** (Kaggle).

**Informational only. Not a medical device and not a diagnosis.**

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Pick **"Sample: 7 days with activity & sleep"** and press **Run analysis**. The
sample is one real, anonymised participant, so nothing needs downloading.

## How it works (plain language)

1. **Cleaning** - removes duplicates and impossible values, puts readings on a
   regular clock, smooths jitter, and scores data quality out of 100.
2. **Context** - joins heart rate with activity level, steps and sleep, either
   from the user's Fitbit files or, if only heart rate is uploaded, estimated by
   the trained activity model.
3. **Numbers** - resting HR (measured while still and awake), heart rate during
   sleep, heart rate by activity level, peaks, daily and hour-of-day patterns,
   exercise episodes with recovery.
4. **Machine learning**
   - *Supervised activity model* (gradient boosting): predicts sedentary / light /
     active for every minute from heart rate alone. It learned from 471,487
     minutes that carry real Fitbit activity labels.
   - *Anomaly detection* (Isolation Forest or Local Outlier Factor): flags
     10-minute windows where heart rate is unusual **for what the person was
     doing**, so workouts are not flagged just for being high.
   - *Clustering* (KMeans): groups windows into behaviour patterns.
5. **Interpretation & report** - plain sentences, gentle alerts, disclaimer, PDF
   and CSV downloads.

## Results (details in `VALIDATION.md`)

| What | Result |
|---|---|
| Data | 15 participants, 3.6 M heart-rate readings, up to 43 days each, both Kaggle export periods |
| Activity model, tested on unseen people | 81% accuracy (vs 73% always guessing "sedentary" and 76% for a heart-rate threshold rule), ROC-AUC 0.98 for active vs not |
| Resting HR from heart rate only vs from real activity files | 1.3 bpm average difference |
| Anomaly detector, hidden-anomaly benchmark | ROC-AUC 0.96-0.98, recall 0.76-0.88 |
| Activity-aware anomaly mode on real data | share of flags that are just exercise drops from 40% to 25% |

## Using the backend from your own frontend

```python
from hr_backend import analyze, fitbit

result = analyze(hr_df)                                   # heart rate only
result = analyze(hr_df,                                    # with context files
                 intensity_df=fitbit.load_intensity([...]),
                 steps_df=fitbit.load_steps([...]),
                 sleep_df=fitbit.load_sleep([...]))

result.ok, result.error      # the pipeline never throws to the UI
result.quality               # score, coverage, warnings, participant id
result.stats                 # resting_hr, overnight_low_hr, mean, max, hrv_proxy ...
result.context               # hr_by_activity, sleep HR, high-HR-while-still episodes
result.minutes               # per-minute HR + activity + sleep (for charts)
result.clean                 # cleaned, smoothed series + moving averages
result.daily, result.circadian, result.peaks
result.anomalies, result.clusters, result.activity_model
result.interpretation        # summary_text, trend_text, alerts[], disclaimer
```

Uploaded companion files can be sorted automatically with
`fitbit.read_companions([df, df, ...])`.

## Reproducing everything from the Kaggle download

Download **FitBit Fitness Tracker Data** from Kaggle and unzip it. Then:

```bash
pip install -r requirements-dev.txt
F1="archive/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16"
F2="archive/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16"
python build_dataset.py "$F1" "$F2"        # merge both periods -> data/
python train_activity_model.py             # evaluate + retrain the model
python validate_report.py "$F1" "$F2"      # regenerate VALIDATION numbers
python test_pipeline.py                    # self-test (needs only sample_data/)
```

## Project layout

```
app.py                     Streamlit app (7 tabs)
hr_backend/
  preprocessing.py         cleaning, resampling, quality score
  fitbit.py                Fitbit file loaders, minute-level merge, upload detection
  context.py               activity/sleep-aware stats, residual features
  activity_model.py        supervised model: features, evaluation, training
  models/                  trained model + metadata (metrics, versions)
  features.py              statistics, rolling/daily/hourly, exercise episodes
  ml.py                    anomaly detection, clustering, classifier scaffold
  interpretation.py        alerts, plain-language summary, disclaimer
  evaluation.py            hidden-anomaly benchmark
  report.py                PDF + CSV export
  pipeline.py              analyze() - the single entrypoint
  synthetic.py             synthetic test data
sample_data/               7 real days of one participant (HR, activity, steps, sleep)
build_dataset.py  train_activity_model.py  validate_report.py  test_pipeline.py
download_data.py           heart-rate-only fallback download (OpenML mirror)
VALIDATION.md  DEPLOYMENT.md  huggingface_README.md
```

## Limitations

- Not a medical device; outputs are informational.
- Heart-rate variability is approximated from BPM; true HRV needs beat-to-beat data.
- Training data is 15 adults from 2016 on Fitbit devices; other watches and
  populations may behave differently.
- Light activity is the hardest class for the model (F1 0.65).
- Anomaly accuracy is measured with hidden fake anomalies because the public
  data has no doctor-confirmed labels.

## Future scope

Wearable APIs (live data), true HRV from beat-to-beat intervals, ECG
compatibility, sleep-stage estimation from heart rate, cloud storage,
multi-sensor fusion, and LLM-written reports. `analyze()` returns one typed
result object, so new modules can be added without changing the frontend contract.

## License

MIT for the code. The dataset is CC0 (public domain).
