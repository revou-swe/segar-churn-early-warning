"""Model candidates and the business-facing evaluation metrics.

The headline metric is NOT accuracy or AUC. The field team can make
VISIT_CAPACITY retention visits a month, so what matters is:

    revenue-at-risk captured @K =
        annualised revenue of outlets that actually churned AND were in our top-K list
        ---------------------------------------------------------------------------
        annualised revenue of all outlets that actually churned

i.e. "of the money we were about to lose, how much did the list point us to?"
AUC / PR-AUC / Brier are reported as supporting diagnostics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C

FEATURES = C.NUMERIC_FEATURES + C.CATEGORICAL_FEATURES
# heavy-tailed counts / amounts get a log transform for the linear model
LOG_FEATURES = ["recency_days", "revenue_90d_m", "avg_order_value_k", "orders_90d", "complaints_90d",
                "distance_km", "days_since_visit", "tenure_months", "avg_days_late_90d"]


def make_logistic() -> Pipeline:
    other = [c for c in C.NUMERIC_FEATURES if c not in LOG_FEATURES]
    pre = ColumnTransformer([
        ("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()), LOG_FEATURES),
        ("num", StandardScaler(), other),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), C.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.5, max_iter=3000))])


def make_gbm() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", C.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATEGORICAL_FEATURES),
    ])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=80, l2_regularization=1.0, random_state=C.SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


MODEL_FACTORIES = {"logistic_regression": make_logistic, "gradient_boosting": make_gbm}

# Baselines = what the sales team does today, written as a score (higher = visit first)
BASELINES = {
    # Today's practice: reps chase whoever hasn't ordered for longest.
    "baseline_recency": lambda df: df["recency_days"].to_numpy(dtype=float),
    # Sales-manager heuristic: anyone silent >30 days, biggest accounts first.
    "baseline_rule_30d_by_revenue": lambda df: ((df["recency_days"] > 30) * (1e6 + df["revenue_at_stake_m"])
                                               + df["recency_days"] / 1e3).to_numpy(dtype=float),
}


def priority_score(prob: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Expected annualised revenue lost = P(churn) x revenue at stake (IDR m)."""
    return prob * df["revenue_at_stake_m"].to_numpy()


def visit_list_mask(df: pd.DataFrame, prob: np.ndarray, k: int = C.VISIT_CAPACITY,
                    reserved: int = C.RESERVED_SMALL_SLOTS, floor: float = 0.0) -> np.ndarray:
    """The visit-list policy, applied per snapshot. Returns a boolean mask aligned with df.

    v1.0: k slots by expected revenue lost (P x revenue).
    v1.1: (k - reserved) slots by expected revenue lost + `reserved` slots for the
          highest-probability Warung outlets not already on the list.
    v1.2: as v1.1, but outlets below a minimum churn probability (`floor`) only enter the
          revenue lane if there are not enough eligible outlets.  <- deployed
    """
    d = df[["snapshot", "outlet_type", "revenue_at_stake_m"]].copy()
    ev = prob * d.revenue_at_stake_m.to_numpy()
    d["prob"], d["ev"], d["pos"] = prob, np.where(prob >= floor, ev, ev * 1e-6), np.arange(len(d))
    chosen = []
    for _, g in d.groupby("snapshot"):
        main = g.nlargest(k - reserved, "ev")
        rest = g[~g.pos.isin(main.pos) & (g.outlet_type == "Warung")].nlargest(reserved, "prob")
        chosen += main.pos.tolist() + rest.pos.tolist()
    mask = np.zeros(len(d), dtype=bool)
    mask[chosen] = True
    return mask


def mask_metrics(df: pd.DataFrame, mask: np.ndarray) -> dict:
    d = df[["snapshot", "churned", "revenue_at_stake_m", "outlet_type"]].assign(sel=mask)
    rows = []
    for _, g in d.groupby("snapshot"):
        s = g[g.sel]
        lost = (g.churned * g.revenue_at_stake_m).sum()
        w = g[g.outlet_type == "Warung"]
        rows.append({"revenue_captured_at_k": (s.churned * s.revenue_at_stake_m).sum() / lost,
                     "precision_at_k": s.churned.mean(),
                     "churners_caught_at_k": s.churned.sum() / max(1, g.churned.sum()),
                     "warung_churners_caught": (w.sel & (w.churned == 1)).sum() / max(1, w.churned.sum())})
    return {k: float(round(v, 4)) for k, v in pd.DataFrame(rows).mean().items()}


def topk_metrics(df: pd.DataFrame, score: np.ndarray, k: int = C.VISIT_CAPACITY) -> dict:
    """Per-snapshot top-K metrics, averaged across snapshots (each month is a separate decision)."""
    d = df[["snapshot", "churned", "revenue_at_stake_m"]].copy()
    d["score"] = score
    rows = []
    for _, g in d.groupby("snapshot"):
        top = g.nlargest(k, "score")
        lost = (g.churned * g.revenue_at_stake_m).sum()
        rows.append({
            "revenue_captured_at_k": (top.churned * top.revenue_at_stake_m).sum() / lost if lost else np.nan,
            "precision_at_k": top.churned.mean(),
            "churners_caught_at_k": top.churned.sum() / max(1, g.churned.sum()),
            "revenue_flagged_and_lost_idr_m": (top.churned * top.revenue_at_stake_m).sum(),
        })
    return pd.DataFrame(rows).mean().to_dict()


def full_metrics(df: pd.DataFrame, rank_score: np.ndarray, prob: np.ndarray | None = None) -> dict:
    m = topk_metrics(df, rank_score)
    s = prob if prob is not None else rank_score
    m["roc_auc"] = roc_auc_score(df.churned, s)
    m["pr_auc"] = average_precision_score(df.churned, s)
    if prob is not None:
        m["brier"] = brier_score_loss(df.churned, prob)
        m["mean_predicted"] = float(np.mean(prob))
        m["actual_rate"] = float(df.churned.mean())
    return {k: float(round(v, 4)) for k, v in m.items()}


def gains_curve(df: pd.DataFrame, score: np.ndarray, ks=range(0, 1001, 25)) -> pd.DataFrame:
    """Revenue-at-risk captured as a function of list length K (averaged over snapshots)."""
    return pd.DataFrame([{"k": k, "captured": topk_metrics(df, score, k)["revenue_captured_at_k"] if k else 0.0}
                         for k in ks])
