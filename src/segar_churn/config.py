"""Central configuration. Every number that drives a business decision lives here,
so a reviewer can find (and challenge) it in one place."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
MODELS = ROOT / "models"
APP = ROOT / "app"

SEED = 42

# ---- Data window -----------------------------------------------------------
START_MONTH = "2024-10"          # first month of transaction history
N_MONTHS = 24                    # Oct-2024 .. Sep-2026
SCORING_MONTH = "2026-09"        # "today": the model scores outlets as of 30-Sep-2026

# ---- Business definitions (agreed with Head of Sales, see reports/01) -------
ACTIVE_LOOKBACK_MONTHS = 3       # an outlet is "active" if it ordered in the last 3 months
CHURN_HORIZON_MONTHS = 2         # churn = zero orders in the next 2 months (≈60 days)
REVENUE_ANNUALISATION = 4        # revenue at stake = last-90-day revenue x 4

# ---- Operating constraint -----------------------------------------------------
VISIT_CAPACITY = 200             # extra retention visits the field team can make per month
RESERVED_SMALL_SLOTS = 40        # of which reserved for Warung outlets ranked by churn probability
                                 # (added after stakeholder pushback, see reports/06)
FLOOR_GRID = [0.0, 0.03, 0.05, 0.08, 0.10]   # candidate minimum churn probability for the revenue lane
FLOOR_MAX_REVENUE_GIVEUP = 0.03  # pick the highest floor that loses <=3pp revenue capture on validation

# ---- Temporal validation split (snapshot month index, 0 = START_MONTH) -------
FIRST_SNAPSHOT = 6               # need 6 months of history for trend features
TRAIN_LAST = 15                  # snapshots 6..15  -> train    (Apr-2025 .. Jan-2026)
VALID_LAST = 18                  # snapshots 16..18 -> validate (Feb-2026 .. Apr-2026)
TEST_LAST = 21                   # snapshots 19..21 -> test     (May-2026 .. Jul-2026)
                                 # label needs 2 future months, so 21 = Jul-2026 is the last labelled snapshot
SCORING_SNAPSHOT = 23            # Sep-2026: live scoring, no label yet

# ---- Risk tiers on calibrated probability ------------------------------------
TIER_HIGH = 0.35
TIER_MEDIUM = 0.15

NUMERIC_FEATURES = [
    "recency_days",
    "orders_90d",
    "order_trend",
    "revenue_90d_m",
    "revenue_trend",
    "avg_order_value_k",
    "sku_breadth_90d",
    "sku_trend",
    "avg_days_late_90d",
    "share_late_14_90d",
    "avg_discount_pct_90d",
    "visits_60d",
    "days_since_visit",
    "complaints_90d",
    "tenure_months",
    "distance_km",
]
CATEGORICAL_FEATURES = ["outlet_type", "region"]

FEATURE_LABELS = {
    "recency_days": "Days since last order",
    "orders_90d": "Orders in last 90 days",
    "order_trend": "Order frequency trend",
    "revenue_90d_m": "Revenue last 90 days",
    "revenue_trend": "Revenue trend",
    "avg_order_value_k": "Average order value",
    "sku_breadth_90d": "Distinct SKUs bought",
    "sku_trend": "SKU range shrinking",
    "avg_days_late_90d": "Paying invoices late",
    "share_late_14_90d": "Share of invoices >14 days late",
    "avg_discount_pct_90d": "Discount dependence",
    "visits_60d": "Few sales-rep visits",
    "days_since_visit": "Long time since rep visit",
    "complaints_90d": "Recent complaints / returns",
    "tenure_months": "New partner (short tenure)",
    "distance_km": "Far from depot",
    "outlet_type": "Outlet type",
    "region": "Region",
}
