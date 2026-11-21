# Student Pass/Fail Predictor

Predicts whether a student will **pass or fail** using modern gradient-boosting models
(LightGBM and XGBoost) trained on large Kaggle datasets, with automatic hyperparameter
tuning (Optuna).

## What it does

1. Downloads a student dataset from Kaggle.
2. Converts the result column into Pass (1) / Fail (0).
3. Splits the data 80/20 **before** any preprocessing, so there is no data leakage.
4. Tunes LightGBM with Optuna using 5-fold cross-validation.
5. Trains and compares four models:
   - Logistic regression (simple baseline, for comparison)
   - Tuned LightGBM
   - XGBoost
   - Ensemble of LightGBM + XGBoost (soft voting)
6. Picks the best model by cross-validated ROC-AUC and reports accuracy, F1 and ROC-AUC on the test set.
7. Saves the model, a metrics file and a feature-importance chart in `models/`.
## Model Performance

The project compares four machine learning approaches:

- Logistic Regression (Baseline)
- Tuned LightGBM
- XGBoost
- Soft Voting Ensemble (LightGBM + XGBoost)

### Best Model: Soft Voting Ensemble

The ensemble achieved the following results on the held-out test dataset:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 94.81% |
| F1 Score | 95.67% |
| ROC-AUC | 98.16% |
| Cross-validation ROC-AUC | 98.03% |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Fail | 0.96 | 0.91 | 0.94 |
| Pass | 0.94 | 0.97 | 0.96 |

The model uses Optuna for hyperparameter optimization of LightGBM and compares the performance of multiple classifiers before selecting the best model using cross-validation.

## Datasets

| Preset | Kaggle source | Size | Pass / fail rule |
|---|---|---|---|
| `academic_success` (default) | Competition `playground-series-s4e6` | ~76,000 students | Graduate = pass, Dropout = fail (Enrolled rows dropped) |
| `performance_factors` | Dataset `lainguyn123/student-performance-factors` | ~6,600 students | Exam score >= 65 = pass |

Presets live in `src/config.py`. You can change the pass mark or add your own dataset there.
For the competition dataset, open its Kaggle page and click **Join Competition** once, or the download is refused.

## Setup

```bash
pip install -r requirements.txt
```

Kaggle API key: kaggle.com → Settings → API → **Create New Token**. Move the downloaded
`kaggle.json` to `~/.kaggle/` (Windows: `C:\Users\<you>\.kaggle\`).

## Run

```bash
python src/download_data.py --dataset academic_success
python src/train.py --dataset academic_success --trials 30
python src/predict.py --dataset academic_success --csv new_students.csv
```

More `--trials` usually gives slightly better accuracy but takes longer.

## Run in Google Colab

```python
!git clone <your-repo-url>
%cd <repo-folder>
!pip install -q -r requirements.txt
from google.colab import files; files.upload()   # upload kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!python src/download_data.py --dataset academic_success
!python src/train.py --dataset academic_success --trials 30
```

## Project structure

```
src/
  config.py         dataset presets
  download_data.py  Kaggle download
  prepare.py        loading + pass/fail label
  train.py          tuning, training, comparison, saving
  predict.py        predictions for new students
models/             created after training (ignored by git)
data/               created after download (ignored by git)
```

## Ideas to improve accuracy further

- Increase `--trials` to 100+.
- Add new features, e.g. pass rate of first-semester units (approved / enrolled).
- Tune XGBoost with Optuna the same way as LightGBM.
- Add CatBoost, which handles text categories well, to the ensemble.
