# Deployment (Objective 6)

Standard single-file Streamlit app with exact-pinned `requirements.txt`.
Tested on Python 3.13; Python 3.11 or newer is required.

## A. Streamlit Community Cloud

1. Push this folder to a GitHub repository (the `.gitignore` already leaves out
   the large `data/` folder; `sample_data/` and `hr_backend/models/` are included
   and are needed).
2. Go to https://share.streamlit.io and choose **Create app**.
3. Select the repository and branch, set **Main file path** to `app.py`, and under
   **Advanced settings** pick Python 3.13 (or the newest offered, 3.11 or later).
4. Deploy.

## B. Hugging Face Spaces

1. Create a new Space and choose the **Streamlit** SDK.
2. Upload every file in this folder (or `git push` to the Space).
3. The Space's `README.md` must start with the YAML header in
   `huggingface_README.md`. Rename that file to `README.md` inside the Space and
   keep the project README under another name, such as `PROJECT_README.md`.

## If installation fails on a platform

The pins are the exact versions tested together. If a platform cannot install
one of them:

1. Loosen that pin (for example `pandas>=2.2`).
2. If scikit-learn changes version, the bundled model may not load. The app keeps
   working and simply skips activity estimation for heart-rate-only uploads.
   To restore it, retrain in the new environment with
   `python train_activity_model.py` (needs the Kaggle data, see README).

## Notes

- No system packages are needed, so no `packages.txt`.
- The upload limit is set to 200 MB in `.streamlit/config.toml`. A full 43-day
  Fitbit heart-rate file for one person is about 15 MB.
- Free tiers are CPU-only. A 7-day file analyses in a few seconds and a 43-day
  file in about 10 to 20 seconds.
