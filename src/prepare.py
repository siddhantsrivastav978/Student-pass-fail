"""Load a dataset and turn it into features X and a pass(1)/fail(0) label y."""
import os

import pandas as pd

from config import DATASETS, DATA_DIR


def load(name, csv_path=None):
    cfg = DATASETS[name]
    path = csv_path or os.path.join(DATA_DIR, name, cfg["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python src/download_data.py --dataset {name}")
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in cfg.get("drop_columns", []) if c in df.columns])
    df = df.drop_duplicates()

    target = cfg["target"]
    if "pass_mark" in cfg:
        y = (df[target] >= cfg["pass_mark"]).astype(int)
    else:
        keep = df[target].isin(cfg["pass_values"] + cfg["fail_values"])
        df = df[keep]
        y = df[target].isin(cfg["pass_values"]).astype(int)

    X = df.drop(columns=[target])
    return X.reset_index(drop=True), y.reset_index(drop=True)
