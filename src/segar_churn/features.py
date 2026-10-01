"""Point-in-time feature engineering.

A *snapshot* is "what we knew at the end of month t". Every feature for snapshot t
is computed only from events dated on or before the snapshot date, and the label
only from events strictly after it. tests/test_features.py enforces this
(no-leakage test), because a churn model that peeks at the future looks
brilliant in the notebook and useless on Monday.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def load_clean() -> dict[str, pd.DataFrame]:
    p = C.DATA_PROCESSED
    return {
        "outlets": pd.read_csv(p / "outlets.csv", parse_dates=["onboard_date"]),
        "orders": pd.read_csv(p / "orders.csv", parse_dates=["order_date"]),
        "visits": pd.read_csv(p / "visits.csv", parse_dates=["visit_date"]),
        "complaints": pd.read_csv(p / "complaints.csv", parse_dates=["complaint_date"]),
    }


def month_end(t: int) -> pd.Timestamp:
    return pd.Timestamp(C.START_MONTH + "-01") + pd.DateOffset(months=t + 1) - pd.Timedelta(days=1)


def _window(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Rows with start < date <= end."""
    return df[(df[col] > start) & (df[col] <= end)]


def build_snapshot(tables: dict[str, pd.DataFrame], t: int, with_label: bool = True) -> pd.DataFrame:
    snap = month_end(t)
    out, od, vi, cp = tables["outlets"], tables["orders"], tables["visits"], tables["complaints"]

    hist = od[od.order_date <= snap]
    last90 = _window(od, "order_date", snap - pd.Timedelta(days=90), snap)
    prev90 = _window(od, "order_date", snap - pd.Timedelta(days=180), snap - pd.Timedelta(days=90))
    active_start = month_end(t - C.ACTIVE_LOOKBACK_MONTHS)
    active_ids = _window(od, "order_date", active_start, snap).outlet_id.unique()

    base = out[(out.outlet_id.isin(active_ids)) & (out.onboard_date <= snap)].set_index("outlet_id")
    f = pd.DataFrame(index=base.index)
    f["snapshot"] = snap.strftime("%Y-%m-%d")
    f["snapshot_idx"] = t

    last_order = hist.groupby("outlet_id").order_date.max()
    f["recency_days"] = (snap - last_order.reindex(f.index)).dt.days

    g90, gp = last90.groupby("outlet_id"), prev90.groupby("outlet_id")
    o90 = g90.size().reindex(f.index, fill_value=0)
    op = gp.size().reindex(f.index, fill_value=0)
    r90 = g90.gross_value_idr.sum().reindex(f.index, fill_value=0)
    rp = gp.gross_value_idr.sum().reindex(f.index, fill_value=0)
    f["orders_90d"] = o90
    f["order_trend"] = (o90 - op) / (op + 1)
    f["revenue_90d_m"] = r90 / 1e6
    f["revenue_trend"] = np.clip((r90 - rp) / (rp + 1e5), -1, 3)
    f["avg_order_value_k"] = (r90 / o90.replace(0, np.nan)).fillna(0) / 1e3
    s90 = g90.n_skus.mean().reindex(f.index)
    sp = gp.n_skus.mean().reindex(f.index)
    f["sku_breadth_90d"] = s90.fillna(0)
    f["sku_trend"] = ((s90 - sp) / sp).fillna(0).clip(-1, 2)
    f["avg_days_late_90d"] = g90.days_paid_late.mean().reindex(f.index).fillna(0)
    f["share_late_14_90d"] = last90.assign(l=last90.days_paid_late > 14).groupby("outlet_id").l.mean().reindex(f.index).fillna(0)
    f["avg_discount_pct_90d"] = (g90.discount_pct.mean().reindex(f.index).fillna(0) * 100)

    v60 = _window(vi, "visit_date", snap - pd.Timedelta(days=60), snap)
    f["visits_60d"] = v60.groupby("outlet_id").size().reindex(f.index, fill_value=0)
    last_visit = vi[vi.visit_date <= snap].groupby("outlet_id").visit_date.max().reindex(f.index)
    f["days_since_visit"] = (snap - last_visit).dt.days.fillna(365).clip(upper=365)

    c90 = _window(cp, "complaint_date", snap - pd.Timedelta(days=90), snap)
    f["complaints_90d"] = c90.groupby("outlet_id").size().reindex(f.index, fill_value=0)

    f["tenure_months"] = ((snap - base.onboard_date).dt.days / 30.44).round(1)
    f["distance_km"] = base.distance_km
    for c in ["outlet_name", "outlet_type", "region", "city", "rep_id", "rep_name"]:
        f[c] = base[c]

    f["revenue_at_stake_m"] = f.revenue_90d_m * C.REVENUE_ANNUALISATION

    if with_label:
        fut = _window(od, "order_date", snap, month_end(t + C.CHURN_HORIZON_MONTHS))
        f["churned"] = (~f.index.isin(fut.outlet_id.unique())).astype(int)
    return f.reset_index()


def build_panel(tables: dict[str, pd.DataFrame], first: int, last: int) -> pd.DataFrame:
    return pd.concat([build_snapshot(tables, t) for t in range(first, last + 1)], ignore_index=True)


def main() -> None:
    tables = load_clean()
    panel = build_panel(tables, C.FIRST_SNAPSHOT, C.TEST_LAST)
    panel.to_csv(C.DATA_PROCESSED / "panel.csv", index=False)
    live = build_snapshot(tables, C.SCORING_SNAPSHOT, with_label=False)
    live.to_csv(C.DATA_PROCESSED / "scoring_snapshot.csv", index=False)
    print(f"panel: {len(panel):,} outlet-months, churn rate {panel.churned.mean():.1%}")
    print(panel.groupby("snapshot").churned.agg(["size", "mean"]).round(3).to_string())
    print(f"scoring snapshot ({live.snapshot.iat[0]}): {len(live):,} active outlets")


if __name__ == "__main__":
    main()
