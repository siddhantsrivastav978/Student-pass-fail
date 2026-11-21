"""Train a student pass/fail model.

Compares a simple baseline with tuned gradient-boosting models (LightGBM, XGBoost)
and a soft-voting ensemble, then saves the best one.

Example:
    python src/train.py --dataset academic_success --trials 30
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from config import DATASETS, MODEL_DIR, RANDOM_STATE
from prepare import load

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tree_preprocessor(X):
    """Trees don't need scaling: fill gaps and turn text categories into numbers."""
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.columns.difference(num).tolist()
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat),
    ])


def linear_preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.columns.difference(num).tolist()
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat),
    ])


def tune_lightgbm(X, y, cv, trials):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        )
        model = Pipeline([("prep", tree_preprocessor(X)),
                          ("clf", LGBMClassifier(**params, random_state=RANDOM_STATE, verbose=-1))])
        return cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    print(f"Best LightGBM CV ROC-AUC: {study.best_value:.4f}")
    return study.best_params


def evaluate(model, X_test, y_test):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1": round(float(f1_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
    }


def save_feature_importance(model, X, path):
    clf = model.named_steps["clf"]
    prep = model.named_steps["prep"]
    names = [n.split("__", 1)[-1] for n in prep.get_feature_names_out()]
    imp = pd.Series(clf.feature_importances_, index=names).sort_values().tail(15)
    plt.figure(figsize=(8, 6))
    imp.plot.barh(color="#2a9d8f")
    plt.title("Top 15 features (LightGBM)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="academic_success", choices=DATASETS.keys())
    p.add_argument("--csv", help="Optional path to the CSV if it isn't in data/<dataset>/")
    p.add_argument("--trials", type=int, default=30, help="Optuna tuning trials (more = better but slower)")
    args = p.parse_args()

    X, y = load(args.dataset, args.csv)
    print(f"Rows: {len(X):,}  Features: {X.shape[1]}  Pass rate: {y.mean():.1%}")

    # Split FIRST so the test set never influences training (no data leakage).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"\nTuning LightGBM with {args.trials} trials ...")
    best = tune_lightgbm(X_train, y_train, cv, args.trials)

    lgbm = Pipeline([("prep", tree_preprocessor(X)),
                     ("clf", LGBMClassifier(**best, subsample_freq=1, random_state=RANDOM_STATE, verbose=-1))])
    xgb = Pipeline([("prep", tree_preprocessor(X)),
                    ("clf", XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6,
                                          subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                          eval_metric="logloss", random_state=RANDOM_STATE))])
    models = {
        "baseline_logistic": Pipeline([("prep", linear_preprocessor(X)),
                                       ("clf", LogisticRegression(max_iter=2000))]),
        "lightgbm_tuned": lgbm,
        "xgboost": xgb,
        "ensemble_lgbm_xgb": VotingClassifier([("lgbm", lgbm), ("xgb", xgb)], voting="soft"),
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
        model.fit(X_train, y_train)
        results[name] = {"cv_roc_auc": round(float(cv_auc), 4), **evaluate(model, X_test, y_test)}
        print(f"  {results[name]}")

    # Choose using cross-validation (not the test set), then report its test score.
    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    best_model = models[best_name]
    print(f"\nBest model: {best_name}")
    pred = best_model.predict(X_test)
    print(classification_report(y_test, pred, target_names=["Fail", "Pass"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump({"model": best_model, "columns": X.columns.tolist(), "dataset": args.dataset},
                os.path.join(MODEL_DIR, f"{args.dataset}_model.joblib"))
    with open(os.path.join(MODEL_DIR, f"{args.dataset}_metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "lightgbm_params": best, "results": results}, f, indent=2)
    save_feature_importance(lgbm, X, os.path.join(MODEL_DIR, f"{args.dataset}_feature_importance.png"))
    print(f"\nSaved model, metrics and feature chart to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
