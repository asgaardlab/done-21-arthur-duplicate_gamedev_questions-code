import pandas as pd

read = pd.read_parquet


def save(df, path):
    if not path.exists():
        make_dir(path.parent)
    df.to_parquet(path)


def make_dir(path):
    path.mkdir(parents=True, exist_ok=True)
