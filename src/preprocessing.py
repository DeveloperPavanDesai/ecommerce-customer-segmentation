import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
    )
    return rfm.reset_index()


def prepare_for_clustering(rfm: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, StandardScaler]:
    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    return rfm_log, rfm_scaled, scaler
