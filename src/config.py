from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "online_retail.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure dirs exist when used
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Model artifact names
SCALER_FILENAME = "scaler.joblib"
KMEANS_FILENAME = "kmeans.joblib"
DBSCAN_FILENAME = "dbscan.joblib"
RFM_FILENAME = "rfm_with_segments.parquet"
SEGMENT_MAP_FILENAME = "segment_map.joblib"