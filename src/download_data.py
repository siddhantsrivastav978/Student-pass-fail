"""Download a dataset from Kaggle into data/<name>/.

Needs a Kaggle API token: kaggle.com -> Settings -> API -> Create New Token.
Put kaggle.json in ~/.kaggle/ (Windows: C:\\Users\\<you>\\.kaggle\\).
"""
import argparse
import os
import zipfile

from config import DATASETS, DATA_DIR


def download(name):
    cfg = DATASETS[name]
    from kaggle.api.kaggle_api_extended import KaggleApi  # imported here so --help works without a token

    api = KaggleApi()
    api.authenticate()
    out = os.path.join(DATA_DIR, name)
    os.makedirs(out, exist_ok=True)

    print(f"Downloading {cfg['kaggle_ref']} ...")
    if cfg["kaggle_type"] == "competition":
        api.competition_download_files(cfg["kaggle_ref"], path=out)
    else:
        api.dataset_download_files(cfg["kaggle_ref"], path=out)

    for f in os.listdir(out):
        if f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(out, f)) as z:
                z.extractall(out)
            os.remove(os.path.join(out, f))
    print("Saved to", out, "->", os.listdir(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="academic_success", choices=DATASETS.keys())
    download(p.parse_args().dataset)
