import pandas as pd
from pathlib import Path

from .config import RAW_DATA_PATH


def load_raw(data_path: Path | None = None) -> pd.DataFrame:
    path = data_path or RAW_DATA_PATH
    return pd.read_csv(path, delimiter=",")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    inv = df["InvoiceNo"].astype(str)
    df = df[~inv.str.startswith("C")]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def load_clean(data_path: Path | None = None) -> pd.DataFrame:
    df = load_raw(data_path)
    return clean_transactions(df)
