import joblib
import pandas as pd
from sklearn.cluster import KMeans

from .config import (
    MODELS_DIR,
    PROCESSED_DIR,
    SCALER_FILENAME,
    KMEANS_FILENAME,
    RFM_FILENAME,
    SEGMENT_MAP_FILENAME,
)
from .data_loader import load_clean
from .preprocessing import compute_rfm, prepare_for_clustering

DEFAULT_SEGMENT_MAP = {
    0: "High Value",
    1: "Loyal",
    2: "At Risk",
    3: "Low Value",
}


def train_and_save(
    n_clusters: int = 4,
    random_state: int = 42,
    save_rfm: bool = True,
) -> pd.DataFrame:
    """Train KMeans, assign segments, save scaler, model, segment map and RFM parquet."""
    df = load_clean()
    rfm = compute_rfm(df)
    _, rfm_scaled, scaler = prepare_for_clustering(rfm)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    rfm["Segment"] = rfm["Cluster"].map(DEFAULT_SEGMENT_MAP)

    joblib.dump(scaler, MODELS_DIR / SCALER_FILENAME)
    joblib.dump(kmeans, MODELS_DIR / KMEANS_FILENAME)
    joblib.dump(DEFAULT_SEGMENT_MAP, MODELS_DIR / SEGMENT_MAP_FILENAME)

    if save_rfm:
        rfm.to_parquet(PROCESSED_DIR / RFM_FILENAME, index=False)

    return rfm


if __name__ == "__main__":
    rfm = train_and_save()
    print("Saved:", MODELS_DIR / SCALER_FILENAME, MODELS_DIR / KMEANS_FILENAME)
    print("Segment counts:\n", rfm["Segment"].value_counts())