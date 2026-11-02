# Changelog

## v0.2.1 - deployment fixes (Nov 2026)

Response to the review of the Smartwatch Heart-Rate Analyzer.

| # | Review point | Status in this repository |
|---|---|---|
| 1 | `sample_data/` missing, Sample options crash | The four sample files are committed, and `.gitignore` does not exclude them. **Added safety nets:** if the files are ever missing, the app shows a warning and uses synthetic data, and `test_pipeline.py` skips the sample checks instead of crashing. |
| 2 | Windows crash in `synthetic.py` (`%-m/%-d` in strftime) | **Fixed.** Timestamps are now built without platform-specific strftime flags, and the output is identical to before. A test checks the format. |
| 3 | Duplicate backend modules, model files and `config.toml` at the project root | Not present here. The only copies are in `hr_backend/` and `.streamlit/`. |
| 4 | Messy `.gitignore` (repeated blocks, personal filename, sample_data excluded) | **Rewritten.** One block per category, no personal files, and a note that `sample_data/` and `hr_backend/models/` must be committed. |
| 5 | Very recent exact version pins | **Replaced with tested ranges**, with only scikit-learn kept exact (the model needs it). Clean installs pass both test suites on Python 3.11, 3.12, 3.13 and 3.14, using both the newest and the oldest allowed packages. The tests found that numpy must be at least 2.0 and Streamlit at least 1.52, so those floors are set. The old exact pins could not install on Python 3.11 at all. |
| 5b | App behaviour when the scikit-learn version differs | **Fixed and tested.** The model loader now refuses a model saved with a different scikit-learn minor version, rather than risk silently wrong predictions. The app keeps running, explains the reason in the Model & Evaluation tab, and treats heart-rate-only uploads as "no activity information". |

Other changes:
- `test_app.py` now returns pass/fail, and fails on crashes, error messages or Streamlit deprecation warnings.
- `DEPLOYMENT.md` now lists the tested Python versions (Python 3.12 recommended), the exact repository name, and a post-deploy checklist.

## v0.2.0 - Oct 2026

Both Fitbit export periods, sleep- and activity-aware analysis, supervised
activity model, activity-aware anomaly detection, full validation (`VALIDATION.md`).
