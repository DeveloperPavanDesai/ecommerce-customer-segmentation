import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from .config import MODELS_DIR, PROCESSED_DIR, SCALER_FILENAME, DBSCAN_FILENAME, RFM_FILENAME
from .data_loader import load_clean
from .preprocessing import compute_rfm, prepare_for_clustering


def train_and_save(eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
    scaler = joblib.load(MODELS_DIR / SCALER_FILENAME)
    df = load_clean()
    rfm = compute_rfm(df)
    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p)
    rfm_scaled = scaler.transform(rfm_log)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    rfm["DBSCAN_Cluster"] = dbscan.fit_predict(rfm_scaled)

    joblib.dump(dbscan, MODELS_DIR / DBSCAN_FILENAME)

    # If KMeans parquet exists, add Cluster and Segment to this RFM
    path = PROCESSED_DIR / RFM_FILENAME
    if path.exists():
        existing = pd.read_parquet(path)
        if "Segment" in existing.columns and "Cluster" in existing.columns:
            rfm = rfm.merge(
                existing[["CustomerID", "Cluster", "Segment"]],
                on="CustomerID",
                how="left",
            )
    rfm.to_parquet(path, index=False)
    return rfm


if __name__ == "__main__":
    rfm = train_and_save()
    print("Saved:", MODELS_DIR / DBSCAN_FILENAME)
    print("DBSCAN_Cluster value counts:\n", rfm["DBSCAN_Cluster"].value_counts())