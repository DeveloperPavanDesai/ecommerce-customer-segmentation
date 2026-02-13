import sys
from pathlib import Path

# Ensure project root is on path when running as app.app
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import pandas as pd
from flask import Flask, jsonify, Response
from flasgger import Swagger

from src.config import MODELS_DIR, PROCESSED_DIR, RFM_FILENAME

app = Flask(__name__)

# Swagger (OpenAPI) config – UI at /apidocs
app.config["SWAGGER"] = {"title": "Customer Segmentation API", "uiversion": 3}
swagger = Swagger(app, template={
    "info": {
        "title": "Customer Segmentation Analytics API",
        "description": "RFM-based customer segments and analytics. Run `python -m src.train_kmeans` first.",
    },
    "host": None,
    "schemes": ["http", "https"],
})

_scaler = None
_kmeans = None
_dbscan = None
_segment_map = None
_rfm_df = None


def get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    return _scaler


def get_kmeans():
    global _kmeans
    if _kmeans is None:
        _kmeans = joblib.load(MODELS_DIR / "kmeans.joblib")
    return _kmeans


def get_dbscan():
    global _dbscan
    path = MODELS_DIR / "dbscan.joblib"
    if _dbscan is None and path.exists():
        _dbscan = joblib.load(path)
    return _dbscan


def get_segment_map():
    global _segment_map
    if _segment_map is None:
        _segment_map = joblib.load(MODELS_DIR / "segment_map.joblib")
    return _segment_map


def get_rfm_df():
    global _rfm_df
    if _rfm_df is None:
        path = PROCESSED_DIR / RFM_FILENAME
        if not path.exists():
            return None
        _rfm_df = pd.read_parquet(path)
    return _rfm_df


@app.route("/")
def index():
    """
    API info and endpoint list
    ---
    tags: [Info]
    responses:
      200:
        description: Service info and list of endpoints
    """
    return jsonify(
        {
            "service": "Customer Segmentation Analytics API",
            "endpoints": {
                "analytics/overview": "GET – segment counts and high-level stats",
                "analytics/segments": "GET – list segment names and cluster stats",
                "analytics/summary": "GET – RFM cluster summary (mean Recency, Frequency, Monetary)",
                "analytics/customer/<customer_id>": "GET – segment and RFM for one customer",
                "health": "GET – health check",
                "ui": "GET – dashboard UI to try endpoints",
                "apidocs": "GET – Swagger (OpenAPI) UI",
            },
        }
    )


@app.route("/analytics/overview")
def analytics_overview():
    """
    Segment counts and high-level stats
    ---
    tags: [Analytics]
    responses:
      200:
        description: Total customers and segment counts
      503:
        description: RFM data not available (run train_kmeans first)
    """
    rfm = get_rfm_df()
    if rfm is None:
        return (
            jsonify(
                {
                    "error": "No processed RFM data. Run: python -m src.train_kmeans",
                }
            ),
            503,
        )
    counts = rfm["Segment"].value_counts().to_dict()
    total = int(rfm["CustomerID"].nunique())
    return jsonify(
        {
            "total_customers": total,
            "segment_counts": counts,
            "segments": list(counts.keys()),
        }
    )


@app.route("/analytics/segments")
def analytics_segments():
    """
    Per-segment counts and mean RFM
    ---
    tags: [Analytics]
    responses:
      200:
        description: List of segments with count, mean_recency, mean_frequency, mean_monetary
      503:
        description: RFM data not available
    """
    rfm = get_rfm_df()
    if rfm is None:
        return jsonify({"error": "No processed RFM data. Run: python -m src.train_kmeans"}), 503
    seg = (
        rfm.groupby("Segment", as_index=False)
        .agg(
            count=("CustomerID", "count"),
            mean_recency=("Recency", "mean"),
            mean_frequency=("Frequency", "mean"),
            mean_monetary=("Monetary", "mean"),
        )
        .round(2)
    )
    return jsonify(seg.to_dict(orient="records"))


@app.route("/analytics/summary")
def analytics_summary():
    """
    RFM cluster summary (mean Recency, Frequency, Monetary per segment)
    ---
    tags: [Analytics]
    responses:
      200:
        description: Mean R/F/M per segment
      503:
        description: RFM data not available
    """
    rfm = get_rfm_df()
    if rfm is None:
        return jsonify({"error": "No processed RFM data. Run: python -m src.train_kmeans"}), 503
    summary = (
        rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(2)
        .reset_index()
    )
    return jsonify(summary.to_dict(orient="records"))


@app.route("/analytics/customer/<customer_id>")
def analytics_customer(customer_id):
    """
    Segment and RFM for one customer
    ---
    tags: [Analytics]
    parameters:
      - name: customer_id
        in: path
        type: number
        required: true
        description: Customer ID (e.g. 17850)
    responses:
      200:
        description: Customer segment, cluster, Recency, Frequency, Monetary (and DBSCAN_Cluster if present)
      400:
        description: customer_id must be numeric
      404:
        description: Customer not found
      503:
        description: RFM data not available
    """
    rfm = get_rfm_df()
    if rfm is None:
        return jsonify({"error": "No processed RFM data. Run: python -m src.train_kmeans"}), 503
    try:
        cid = float(customer_id)
    except ValueError:
        return jsonify({"error": "customer_id must be numeric"}), 400
    row = rfm[rfm["CustomerID"] == cid]
    if row.empty:
        return jsonify({"error": f"Customer {customer_id} not found"}), 404
    r = row.iloc[0]
    out = {
        "CustomerID": float(r["CustomerID"]),
        "Segment": r["Segment"],
        "Cluster": int(r["Cluster"]),
        "Recency": float(r["Recency"]),
        "Frequency": float(r["Frequency"]),
        "Monetary": float(r["Monetary"]),
    }
    if "DBSCAN_Cluster" in rfm.columns:
        out["DBSCAN_Cluster"] = int(r["DBSCAN_Cluster"])
    return jsonify(out)


@app.route("/ui")
def dashboard_ui():
    """
    Simple dashboard UI to try API endpoints
    ---
    tags: [Info]
    responses:
      200:
        description: HTML dashboard page
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Segmentation API – Dashboard</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 0 auto; padding: 1.5rem; background: #0f0f12; color: #e4e4e7; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    a { color: #7c9cff; }
    .card { background: #18181b; border-radius: 8px; padding: 1rem; margin: 1rem 0; border: 1px solid #27272a; }
    button { background: #3b82f6; color: #fff; border: none; padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; margin-right: 0.5rem; margin-bottom: 0.5rem; }
    button:hover { background: #2563eb; }
    button.secondary { background: #27272a; }
    button.secondary:hover { background: #3f3f46; }
    input[type="number"] { padding: 0.5rem; border-radius: 6px; border: 1px solid #3f3f46; background: #18181b; color: #e4e4e7; width: 120px; }
    pre { background: #0f0f12; padding: 1rem; border-radius: 6px; overflow: auto; font-size: 0.85rem; border: 1px solid #27272a; }
    .out { margin-top: 0.75rem; }
    .links { margin: 1rem 0; }
  </style>
</head>
<body>
  <h1>Customer Segmentation API – Dashboard</h1>
  <p>Try the analytics endpoints below. Data comes from the same API as Swagger.</p>
  <div class="links">
    <a href="/apidocs">→ Swagger UI (OpenAPI)</a>
  </div>

  <div class="card">
    <strong>Health</strong>
    <button onclick="call('/health','out-health')">GET /health</button>
    <div class="out" id="out-health"></div>
  </div>

  <div class="card">
    <strong>Overview</strong>
    <button onclick="call('/analytics/overview','out-overview')">GET /analytics/overview</button>
    <div class="out" id="out-overview"></div>
  </div>

  <div class="card">
    <strong>Segments</strong>
    <button onclick="call('/analytics/segments','out-segments')">GET /analytics/segments</button>
    <div class="out" id="out-segments"></div>
  </div>

  <div class="card">
    <strong>Summary</strong>
    <button onclick="call('/analytics/summary','out-summary')">GET /analytics/summary</button>
    <div class="out" id="out-summary"></div>
  </div>

  <div class="card">
    <strong>Customer by ID</strong>
    <input type="number" id="customerId" placeholder="e.g. 17850" value="17850">
    <button onclick="call('/analytics/customer/'+document.getElementById('customerId').value,'out-customer')">GET customer</button>
    <div class="out" id="out-customer"></div>
  </div>

  <script>
    async function call(path, outId) {
      var el = document.getElementById(outId);
      el.innerHTML = '<pre>Loading...</pre>';
      try {
        var r = await fetch(window.location.origin + path);
        var j = await r.json();
        el.innerHTML = '<pre>' + JSON.stringify(j, null, 2) + '</pre>';
      } catch (e) {
        el.innerHTML = '<pre>Error: ' + e.message + '</pre>';
      }
    }
  </script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


@app.route("/health")
def health():
    """
    Health check
    ---
    tags: [Info]
    responses:
      200:
        description: status, models_loaded, dbscan_loaded, rfm_data_available
    """
    models_ok = (MODELS_DIR / "scaler.joblib").exists() and (MODELS_DIR / "kmeans.joblib").exists()
    dbscan_ok = (MODELS_DIR / "dbscan.joblib").exists()
    rfm_ok = (PROCESSED_DIR / RFM_FILENAME).exists()
    return jsonify(
        {
            "status": "ok",
            "models_loaded": models_ok,
            "dbscan_loaded": dbscan_ok,
            "rfm_data_available": rfm_ok,
        }
    )