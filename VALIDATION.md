# Validation Report

How the app was tested on **real** data, including what works, what doesn't,
and the trade-offs. Every number here can be regenerated with
`python validate_report.py <folder1> <folder2>` (raw output:
`validation_output.md`, `validation_results.json`).

## Data

**FitBit Fitness Tracker Data** (Kaggle, CC0), both export periods
(12 Mar - 11 Apr 2016 and 12 Apr - 12 May 2016), merged and de-duplicated:

| | |
|---|---|
| Heart-rate readings | 3,614,915 (every 5-15 s) |
| Participants with heart rate | 15 |
| Days per participant | up to 43 |
| Minutes with heart rate **and** a Fitbit activity label | 471,487 |
| Participants with sleep logs | 12 |

Activity labels (Fitbit intensity) for those minutes: 73% sedentary, 22% light,
5% moderate or vigorous. In this project, "active" means moderate or vigorous combined.

## 1. Every participant, analysed with real activity and sleep data

All 15 participants ran end to end without errors.

| participant | days | quality | resting HR | sleep HR | HR sedentary | HR active | % active | high-HR-while-still episodes |
|---|---|---|---|---|---|---|---|---|
| 2022484408 | 41.3 | 69 | 62.3 | - | 69.0 | 111.7 | 8.4 | 0 |
| 2026352035 | 37.8 | 55 | 67.5 | 66.1 | 87.2 | - | 0.0 | 0 |
| 2347167796 | 31.3 | 95 | 66.2 | 65.9 | 71.8 | 105.4 | 2.8 | 0 |
| 4020332650 | 41.5 | 73 | 73.7 | 64.7 | 85.3 | 106.5 | 2.1 | 1 |
| 4388161847 | 29.6 | 95 | 56.5 | 55.1 | 60.7 | 97.0 | 3.5 | 1 |
| 4558609924 | 41.7 | 73 | 64.7 | 66.4 | 74.7 | 104.7 | 2.3 | 0 |
| 5553957443 | 41.4 | 94 | 60.1 | 59.9 | 63.4 | 97.0 | 2.9 | 0 |
| 5577150313 | 40.5 | 92 | 56.0 | 51.6 | 61.0 | 108.5 | 9.7 | 0 |
| 6117666160 | 38.5 | 81 | 74.8 | 69.6 | 82.6 | 113.2 | 0.4 | 0 |
| 6391747486 | 3.1 | 61 | 77.5 | 76.2* | 82.7 | 108.1 | 0.9 | 0 |
| 6775888955 | 36.1 | 64 | 75.2 | 76.5 | 90.3 | 115.8 | 12.2 | 2 |
| 6962181067 | 43.5 | 96 | 67.0 | 64.3 | 71.9 | 111.7 | 3.6 | 0 |
| 7007744171 | 35.6 | 80 | 72.8 | 67.4 | 81.0 | 114.6 | 6.4 | 0 |
| 8792009665 | 33.5 | 83 | 66.0 | 63.9 | 72.8 | 108.2 | 0.7 | 1 |
| 8877689391 | 41.3 | 75 | 60.2 | 75.4* | 65.8 | 133.8 | 9.0 | 0 |

\* No sleep log, so sleep was assumed to be 00:00-06:00.

**What this shows.**
- Resting heart rates of 56-78 bpm are realistic.
- Heart rate rises from sedentary to active for every participant, so the activity labels and the merge both behave as expected.
- Participant 8877689391 shows why the sleep assumption is a limitation. Their "sleep" heart rate (75) is higher than their resting heart rate (60), which means they were probably awake or not wearing the watch between 00:00 and 06:00. With a real sleep log, the app uses actual sleep times instead.
- Quality scores are lower for people who wore the watch only part of the time. One participant has readings for just 0.7% of their 38-day span.

## 2. Supervised activity model

**Task.** Predict each minute's activity level (sedentary / light / active) from heart rate alone.

**Testing.** Leave-participants-out (5 folds, split by person). The model is
always scored on people it never saw during training, which is the honest
setting for an app used by new people.

| model | accuracy | balanced accuracy | macro F1 |
|---|---|---|---|
| Always guess "sedentary" | 0.731 | 0.333 | - |
| Heart-rate threshold rule (logistic regression on HR above personal baseline) | 0.757 | 0.742 | 0.665 |
| **Trained model (gradient boosting, class-balanced)** | **0.813** | **0.780** | **0.738** |

ROC-AUC: 0.927 (macro, one-vs-rest); **0.981 for active vs not active**.

| class | precision | recall | F1 | minutes |
|---|---|---|---|---|
| sedentary | 0.939 | 0.837 | 0.885 | 344,811 |
| light | 0.569 | 0.746 | 0.646 | 105,062 |
| active | 0.621 | 0.757 | 0.682 | 21,614 |

**Honest reading.**
- The model clearly beats both baselines.
- Its main weakness is **light** activity, which sits between sitting and exercising in heart-rate terms.
- Active minutes are almost never mistaken for sedentary (78 of 21,614).

## 3. Heart rate only: does the estimated context match the real one?

Most app users will upload only a heart-rate file. To test that case, the model
was retrained 15 times, leaving each participant out once. Each time, the
left-out person's resting HR and time spent active were estimated from heart
rate alone, then compared with the values computed from their real Fitbit
activity files.

**Resting HR: mean absolute difference 1.3 bpm** (worst case 4.1 bpm).

**Time active is rougher.** For some people it is over-estimated. For example,
participant 6117666160 was 0.4% active in reality but 5.9% estimated. The app
therefore tells the user when activity was estimated, and recommends uploading
the activity file for exact numbers.

## 4. Anomaly detection

The public data has no doctor-confirmed anomaly labels, so two checks were run.

### 4a. Hidden-anomaly benchmark

In each real recording we hid 15 spikes, 10 dropouts and 5 sustained stretches
of raised heart rate. That is about 31 anomalous 10-minute windows per person.
We then measured how many the detector found. Results are averaged over the 5
participants with the most data and 3 random hidings each, with the detector
set to flag 5% of windows.

| method | features | precision | recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Isolation Forest | plain | 0.132 | 0.807 | 0.223 | 0.967 | 0.376 |
| Isolation Forest | activity-aware | 0.126 | 0.760 | 0.212 | 0.955 | 0.266 |
| Local Outlier Factor | plain | 0.137 | 0.881 | 0.233 | 0.976 | 0.192 |
| Local Outlier Factor | activity-aware | 0.132 | 0.814 | 0.223 | 0.958 | 0.291 |

**Why precision is low.** These recordings are long (2,000-6,000 windows), so
flagging 5% means about 200 flags against about 31 hidden anomalies. The highest
precision any detector could reach at this setting is about **0.155**. The
detectors reach 0.13, about 85% of that ceiling. Lowering the "expected anomaly
fraction" in the sidebar raises precision.

### 4b. What gets flagged on real data (no hidden anomalies)

Share of flagged windows, averaged over all 15 participants:

| | plain | activity-aware |
|---|---|---|
| flagged windows that were during movement | 39.7% | **25.3%** |
| flagged windows while sitting still and awake | 9.8% | **20.2%** |

### The trade-off, stated plainly

- The **plain** detector is slightly better at catching artificial spikes.
  However, about 40% of what it flags on real data is ordinary exercise.
- The **activity-aware** detector judges each window against what is normal for
  that activity level. It cuts exercise flags by about a third and doubles flags
  where heart rate was high while the person was still. Those are the moments a
  user actually wants to see. It gives up about 0.01-0.02 ROC-AUC on the
  artificial benchmark to do this.
- The app defaults to **activity-aware**, and the sidebar checkbox switches to plain.

**A design change that came out of this testing.** The first activity-aware
version still included variability features (std, range, slope). On a 7-day
check, 60% of its flags were still during movement, because heart rate varies
more whenever someone moves. Those features were replaced with a single
"largest sudden jump" feature. Exercise raises heart rate gradually, while
glitches and abrupt events jump. That brought movement flags down to 45% on the
same week, and to 25% across everyone. The jump feature also improved the plain
detector.

## Known limitations

- Not a medical device; nothing here is a diagnosis.
- Heart-rate variability is approximated from BPM, not measured beat-to-beat.
- Without a sleep file, sleep is assumed to be 00:00-06:00. Section 1 shows how that can be wrong.
- The training data is 15 adults from 2016 wearing Fitbits, all recruited through Amazon Mechanical Turk. Other watches, ages and health conditions may behave differently.
- Anomaly accuracy is measured with artificial anomalies. Real clinical accuracy is unknown without labelled clinical data.
