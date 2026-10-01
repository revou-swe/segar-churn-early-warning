"""The most important test in the repo: features must not see the future."""
import numpy as np
import pandas as pd

from segar_churn import config as C
from segar_churn.features import build_snapshot, month_end

T = 12


def _truncate(tables, cutoff):
    t = dict(tables)
    t["orders"] = tables["orders"][tables["orders"].order_date <= cutoff]
    t["visits"] = tables["visits"][tables["visits"].visit_date <= cutoff]
    t["complaints"] = tables["complaints"][tables["complaints"].complaint_date <= cutoff]
    return t


def test_no_future_leakage(clean_small):
    full = build_snapshot(clean_small, T, with_label=False)
    past_only = build_snapshot(_truncate(clean_small, month_end(T)), T, with_label=False)
    cols = C.NUMERIC_FEATURES + ["outlet_id"]
    pd.testing.assert_frame_equal(full[cols].reset_index(drop=True), past_only[cols].reset_index(drop=True))


def test_label_matches_definition(clean_small):
    snap = build_snapshot(clean_small, T)
    od = clean_small["orders"]
    fut = od[(od.order_date > month_end(T)) & (od.order_date <= month_end(T + C.CHURN_HORIZON_MONTHS))]
    expected = (~snap.outlet_id.isin(fut.outlet_id)).astype(int)
    assert (snap.churned.to_numpy() == expected.to_numpy()).all()


def test_population_is_active_outlets_only(clean_small):
    snap = build_snapshot(clean_small, T)
    assert (snap.recency_days <= 92).all()
    assert snap.outlet_id.is_unique


def test_features_are_finite(clean_small):
    snap = build_snapshot(clean_small, T)
    assert np.isfinite(snap[C.NUMERIC_FEATURES].to_numpy(dtype=float)).all()
