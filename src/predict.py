"""Predict pass/fail for new students.

    python src/predict.py --dataset academic_success --csv new_students.csv
The CSV must have the same columns used in training (the target column is not needed).
"""
import argparse
import os

import joblib
import pandas as pd

from config import MODEL_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="academic_success")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="predictions.csv")
    args = p.parse_args()

    bundle = joblib.load(os.path.join(MODEL_DIR, f"{args.dataset}_model.joblib"))
    df = pd.read_csv(args.csv)
    missing = set(bundle["columns"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    prob = bundle["model"].predict_proba(df[bundle["columns"]])[:, 1]
    df["pass_probability"] = prob.round(3)
    df["prediction"] = ["Pass" if p >= 0.5 else "Fail" for p in prob]
    df.to_csv(args.out, index=False)
    print(df[["pass_probability", "prediction"]].head(10))
    print(f"\nSaved {len(df)} predictions to {args.out}")


if __name__ == "__main__":
    main()
