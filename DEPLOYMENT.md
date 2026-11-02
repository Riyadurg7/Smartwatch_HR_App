# Deployment (Objective 6)

Single-file Streamlit app. Repository: https://github.com/Riyadurg7/Smartwatch_HR_App

## Supported Python versions

Clean installs of `requirements.txt` were tested on **Python 3.11, 3.12, 3.13
and 3.14**, with the newest allowed packages and with the oldest allowed ones.
Both `test_pipeline.py` and `test_app.py` pass in every case.
**Use Python 3.12** (the Streamlit Cloud default) unless you have a reason not to.

Python 3.10 and older will not work.

## Before you deploy - run the tests

```bash
pip install -r requirements.txt
python test_pipeline.py     # backend checks, ends with ALL CHECKS PASSED
python test_app.py          # clicks through the app, ends with APP CHECKS PASSED
```

## A. Streamlit Community Cloud

1. Make sure the GitHub repository contains `sample_data/` (4 CSV files) and
   `hr_backend/models/` (2 files). Both are committed already.
2. Go to https://share.streamlit.io, sign in with GitHub, and choose **Create app**.
3. Choose "Deploy a public app from GitHub". Repository:
   `Riyadurg7/Smartwatch_HR_App`, branch: `main`, main file path: `app.py`.
4. Open **Advanced settings** and set **Python version** to **3.12**. Save.
5. Click **Deploy**. The first build takes a few minutes.
6. Open the link and check:
   - "Sample: 7 days with activity & sleep" runs, and all 7 tabs show content.
   - "Sample: heart rate only" runs, and the Overview says activity was estimated by the model.
   - "Synthetic (generated)" runs.
   - "Upload my files" works with a Fitbit `heartrate_seconds_merged.csv`.
   - The PDF download in "Report & Export" opens.

## B. Hugging Face Spaces (backup)

1. Create a new Space and choose the **Streamlit** SDK.
2. Upload every file from the repository, including `sample_data/` and `hr_backend/models/`.
3. The Space's `README.md` must start with the YAML header in
   `huggingface_README.md`. Rename that file to `README.md` inside the Space
   and keep the project README under another name, such as `PROJECT_README.md`.

## If something goes wrong

- **Build fails while installing packages:** check the Python version in
  Advanced settings (choose 3.12) and read the first error in the build log.
- **"The activity model is not available here"** in the Model & Evaluation tab:
  the platform installed a different scikit-learn version. The app still works,
  but heart-rate-only uploads skip activity estimation. Make sure
  `scikit-learn==1.9.1` is installed, or retrain the model in that environment
  with `python train_activity_model.py` (needs the Kaggle data, see README).
- **"Sample files are missing"** warning: `sample_data/` was not uploaded. The
  app falls back to synthetic data until it is added.

## Notes

- No system packages are needed, so no `packages.txt`.
- The upload limit is 200 MB (`.streamlit/config.toml`). A 43-day Fitbit
  heart-rate file for one person is about 15 MB.
- A 7-day file analyses in a few seconds on the free tier, and a 43-day file
  in about 10 to 20 seconds.
