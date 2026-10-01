"""Monthly scoring run: produces the visit list, the dashboard data, and drift checks.

Usage:  python -m segar_churn.score            (scores config.SCORING_SNAPSHOT)
Output: outputs/visit_list_<date>.csv          -> what sales managers act on
        outputs/scored_outlets_<date>.csv      -> every active outlet with probability + reasons
        outputs/drift_report_<date>.csv        -> PSI per feature vs training data
        app/dashboard_data.json                -> consumed by app/build_dashboard.py
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .explain import reason_codes
from .modeling import FEATURES, visit_list_mask

ACTIONS = {
    "recency_days": "Call this week to confirm next order; check shelf stock",
    "orders_90d": "Call this week to confirm next order; check shelf stock",
    "order_trend": "Ask what changed: new supplier, cash flow, or footfall?",
    "revenue_trend": "Ask what changed: new supplier, cash flow, or footfall?",
    "sku_breadth_90d": "Range review on visit: check for competitor SKUs on shelf",
    "sku_trend": "Range review on visit: check for competitor SKUs on shelf",
    "avg_days_late_90d": "Agree a payment plan with Finance before pushing new stock",
    "share_late_14_90d": "Agree a payment plan with Finance before pushing new stock",
    "visits_60d": "Book a rep visit within 7 days",
    "days_since_visit": "Book a rep visit within 7 days",
    "complaints_90d": "Close the open complaint first (service recovery)",
    "avg_discount_pct_90d": "Offer loyalty bundle instead of deeper discount",
    "tenure_months": "New-partner check-in: reorder app + credit terms walkthrough",
    "region": "Competitor watch: secure top-10 SKU availability",
    "outlet_type": "Standard retention call",
    "distance_km": "Combine with nearby route; offer scheduled delivery day",
    "avg_order_value_k": "Standard retention call",
    "revenue_90d_m": "Account manager call: review annual terms",
}


def phrase(f: str, r: pd.Series, ref: pd.DataFrame) -> str:
    med = ref[f].median() if f in C.NUMERIC_FEATURES else None
    match f:
        case "recency_days": return f"No order for {int(r[f])} days"
        case "orders_90d": return f"Only {int(r[f])} orders in 90 days (typical healthy: {med:.0f})"
        case "order_trend": return f"Order frequency {r[f]:+.0%} vs prior 90 days"
        case "revenue_trend": return f"Revenue {r[f]:+.0%} vs prior 90 days"
        case "sku_breadth_90d": return f"Only {r[f]:.0f} SKUs per order (typical healthy: {med:.0f})"
        case "sku_trend": return f"SKU range {r[f]:+.0%} vs prior 90 days"
        case "avg_days_late_90d": return f"Pays {r[f]:.0f} days late on average"
        case "share_late_14_90d": return f"{r[f]:.0%} of invoices >14 days late"
        case "visits_60d": return f"{int(r[f])} rep visits in 60 days"
        case "days_since_visit": return "No rep visit on record" if r[f] >= 365 else f"No rep visit for {int(r[f])} days"
        case "complaints_90d": return f"{int(r[f])} complaint(s) in 90 days"
        case "avg_discount_pct_90d": return f"Avg discount {r[f]:.1f}%"
        case "tenure_months": return f"New partner ({r[f]:.0f} months)"
        case "region": return f"{r[f]}: regional churn running high"
        case "outlet_type": return f"{r[f]} segment risk"
        case "distance_km": return f"{r[f]:.0f} km from depot"
        case _: return C.FEATURE_LABELS.get(f, f)


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 watch, >0.25 investigate."""
    if not pd.api.types.is_numeric_dtype(expected):
        e = expected.value_counts(normalize=True)
        a = actual.value_counts(normalize=True).reindex(e.index, fill_value=0)
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = pd.cut(expected, edges).value_counts(normalize=True, sort=False)
        a = pd.cut(actual, edges).value_counts(normalize=True, sort=False)
    e, a = e.clip(lower=1e-4), a.clip(lower=1e-4)
    return float(((a - e) * np.log(a / e)).sum())


def main() -> None:
    bundle = joblib.load(C.MODELS / "churn_model.joblib")
    model, ref = bundle["model"], bundle["train_reference"]
    panel = pd.read_csv(C.DATA_PROCESSED / "panel.csv")
    live = pd.read_csv(C.DATA_PROCESSED / "scoring_snapshot.csv")
    date = live.snapshot.iat[0]

    live["churn_probability"] = model.predict_proba(live[FEATURES])[:, 1]
    live["expected_loss_m"] = live.churn_probability * live.revenue_at_stake_m
    metrics = json.loads((C.REPORTS / "metrics.json").read_text())
    floor = metrics["probability_floor"]
    prob = live.churn_probability.to_numpy()
    live["on_visit_list"] = visit_list_mask(live, prob, floor=floor)
    rev_lane = visit_list_mask(live, prob, reserved=0, k=C.VISIT_CAPACITY - C.RESERVED_SMALL_SLOTS, floor=floor)
    live["lane"] = np.where(~live.on_visit_list, "", np.where(rev_lane, "Revenue lane", "Warung lane"))
    live["risk_tier"] = pd.cut(live.churn_probability, [-1, C.TIER_MEDIUM, C.TIER_HIGH, 2],
                               labels=["Low", "Medium", "High"]).astype(str)
    live = live.sort_values(["on_visit_list", "expected_loss_m"], ascending=False).reset_index(drop=True)
    live["priority_rank"] = np.where(live.on_visit_list, np.arange(1, len(live) + 1), np.nan)

    reasons = reason_codes(model, live, panel)
    healthy = panel[panel.churned == 0]
    live["reason_1"], live["reason_2"], live["reason_3"], live["action"] = "", "", "", ""
    reason_feats = []
    for i, rs in enumerate(reasons):
        txt = [phrase(f, live.iloc[i], healthy) for f, _ in rs]
        for j, t in enumerate(txt):
            live.at[i, f"reason_{j+1}"] = t
        live.at[i, "action"] = ACTIONS.get(rs[0][0], "Standard retention call") if rs else "Account review: large account, no single warning sign"
        reason_feats.append([f for f, _ in rs])

    C.OUTPUTS.mkdir(exist_ok=True)
    cols = ["priority_rank", "lane", "outlet_id", "outlet_name", "outlet_type", "region", "city", "rep_id", "rep_name",
            "churn_probability", "risk_tier", "revenue_at_stake_m", "expected_loss_m", "recency_days",
            "reason_1", "reason_2", "reason_3", "action"]
    live[cols].to_csv(C.OUTPUTS / f"scored_outlets_{date}.csv", index=False, float_format="%.4f")
    live[live.on_visit_list][cols].to_csv(C.OUTPUTS / f"visit_list_{date}.csv", index=False, float_format="%.4f")

    # ---------- drift ----------
    train = panel[panel.snapshot_idx <= C.TRAIN_LAST]
    drift = pd.DataFrame([{"feature": f, "label": C.FEATURE_LABELS[f], "psi": round(psi(train[f], live[f]), 4)}
                          for f in FEATURES]).sort_values("psi", ascending=False)
    drift["status"] = pd.cut(drift.psi, [-1, 0.1, 0.25, 99], labels=["Stable", "Watch", "Investigate"]).astype(str)
    drift.to_csv(C.OUTPUTS / f"drift_report_{date}.csv", index=False)

    # ---------- dashboard payload ----------
    evidence = json.loads((C.REPORTS / "evidence.json").read_text())
    hist = panel.groupby(["snapshot", "region"]).agg(rate=("churned", "mean"), n=("churned", "size")).reset_index()
    hist_tot = panel.assign(l=panel.churned * panel.revenue_at_stake_m).groupby("snapshot") \
        .agg(rate=("churned", "mean"), lost=("l", "sum")).reset_index()
    rec = []
    for i, r in live.iterrows():
        rec.append({
            "id": r.outlet_id, "n": r.outlet_name, "t": r.outlet_type, "g": r.region, "c": r.city,
            "rep": f"{r.rep_name} ({r.rep_id})", "p": round(float(r.churn_probability), 4),
            "tier": r.risk_tier, "rev": round(float(r.revenue_at_stake_m), 1), "ev": round(float(r.expected_loss_m), 1),
            "rec": int(r.recency_days), "list": bool(r.on_visit_list), "lane": r.lane,
            "rk": None if np.isnan(r.priority_rank) else int(r.priority_rank),
            "why": [x for x in [r.reason_1, r.reason_2, r.reason_3] if x], "act": r.action,
            "o90": int(r.orders_90d), "sku": round(float(r.sku_breadth_90d), 1), "late": round(float(r.avg_days_late_90d), 1),
            "vis": int(r.visits_60d)})
    t = metrics["test"]
    sel = metrics["selected_model"]
    pol = metrics["test_policy_comparison"]
    dep = pol[metrics["deployed_policy"]]
    payload = {
        "as_of": date, "capacity": C.VISIT_CAPACITY, "reserved": C.RESERVED_SMALL_SLOTS,
        "model": {"name": sel.replace("_", " ").title(), "trained_on": bundle["trained_on"],
                  "auc": t[sel]["roc_auc"], "pr_auc": t[sel]["pr_auc"], "brier": t[sel]["brier"],
                  "captured": dep["revenue_captured_at_k"], "floor": floor,
                  "captured_v10": pol["v1.0_pure_expected_revenue"]["revenue_captured_at_k"],
                  "captured_baseline": pol["today_recency_baseline"]["revenue_captured_at_k"],
                  "precision": dep["precision_at_k"], "precision_baseline": pol["today_recency_baseline"]["precision_at_k"],
                  "baseline_auc": t["baseline_recency"]["roc_auc"], "test_window": metrics["splits"]["test"]["snapshots"]},
        "evidence": {"rate": evidence["churn_rate"], "lost_per_month": evidence["annualised_revenue_lost_per_month_idr_m"],
                     "late": evidence["H1_late_payers"], "region": evidence["H4_region"]["rates"],
                     "ooc": evidence["p_chart_out_of_control"]},
        "history": {"total": hist_tot.round(4).to_dict("records"), "region": hist.round(4).to_dict("records")},
        "drift": drift.head(8).to_dict("records"),
        "outlets": rec,
    }
    C.APP.mkdir(exist_ok=True)
    (C.APP / "dashboard_data.json").write_text(json.dumps(payload, separators=(",", ":")))

    vl = live[live.on_visit_list]
    print(f"scored {len(live):,} active outlets as of {date}")
    print(f"visit list: {len(vl)} outlets, expected annualised loss covered "
          f"Rp {vl.expected_loss_m.sum()/1e3:,.1f} bn of Rp {live.expected_loss_m.sum()/1e3:,.1f} bn")
    print(live.risk_tier.value_counts().to_string())
    print(drift.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
