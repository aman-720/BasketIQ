"""
BasketIQ - Project Configuration
Central config for paths, parameters, and styling.
"""
import os

# ─── PATHS ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
VIZ_DIR = os.path.join(PROJECT_ROOT, "visualizations")
DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")

# ─── ANALYSIS PARAMETERS ─────────────────────────────────────────────
# Association Rules
MIN_SUPPORT = 0.005          # Minimum support threshold
MIN_CONFIDENCE = 0.1         # Minimum confidence
MIN_LIFT = 1.0               # Minimum lift
MAX_ITEMSET_LEN = 3          # Max items in a frequent itemset
TOP_N_RULES = 50             # Top rules to display

# RFM Segmentation
RFM_QUANTILES = 5            # Number of quantile bins for R, F, M scores
N_CLUSTERS = 5               # Number of customer clusters

# ─── VISUALIZATION STYLE ─────────────────────────────────────────────
PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C']
PALETTE_SEQUENTIAL = 'YlOrRd'
PALETTE_DIVERGING = 'RdBu_r'
FIG_DPI = 150
FIG_SIZE = (12, 7)
FIG_SIZE_WIDE = (14, 6)
FIG_SIZE_SMALL = (8, 5)

# Cluster labels (assigned after analysis)
CLUSTER_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "Potential Loyalists",
    3: "At-Risk",
    4: "Hibernating"
}
