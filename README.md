# Django Machine Learning Lab – Vehicle Analytics

This is the project built from `new-exercise/Last exercise - django_ml_lab_manual.pdf`.

## Prerequisites

- Python (recommended: **3.11+**)
- Pip

## Dataset

The app reads:

- `dummy-data/vehicles_ml_dataset.csv`

## Setup (Windows / PowerShell)

From this folder (`new-exercise/django_ml_project/`):

1) Create and activate a virtual environment:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) (Optional) Run migrations (not strictly needed since we don’t use DB models here):

```powershell
python manage.py migrate
```

## Run the app

```powershell
python manage.py runserver
```

Open:

- `http://127.0.0.1:8000/`

## What to test (quick checklist)

### 1) EDA page

- Open `http://127.0.0.1:8000/` (or `http://127.0.0.1:8000/data_exploration`)
- Confirm you see:
  - “Data Exploration” table (first rows)
  - “Statistical Analysis” table (describe output)

### 2) Regression (Price prediction)

- Go to `http://127.0.0.1:8000/regression_analysis`
- Submit the form with any values
- Confirm:
  - A predicted price appears
  - R2 score and comparison table appear

### 3) Classification (Income level)

- Go to `http://127.0.0.1:8000/classification_analysis`
- Submit the form
- Confirm:
  - A predicted income category appears
  - Accuracy and comparison table appear

### 4) Clustering (Client segmentation)

- Go to `http://127.0.0.1:8000/clustering_analysis`
- Submit the form
- Confirm:
  - Predicted price appears
  - Cluster label appears (“Economy / Standard / Premium”)
  - Silhouette score + summary table appear

## Notes about model files

Models are trained automatically the first time you visit each ML page (if the `.pkl` file doesn’t exist yet):

- Regression: `model_generators/regression/regression_model.pkl`
- Classification: `model_generators/classification/classification_model.pkl`
- Clustering bundle (model + label mapping): `model_generators/clustering/clustering_model.pkl`

If you want to retrain from scratch, you can delete those `.pkl` files and reload the pages.

## Troubleshooting

- **Blank/500 error on EDA page**: confirm `dummy-data/vehicles_ml_dataset.csv` exists.
- **Import / package errors**: re-run `pip install -r requirements.txt` inside the activated venv.
- **Port already in use**:

```powershell
python manage.py runserver 8001
```

