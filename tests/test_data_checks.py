import pandas as pd
import pytest

from segar_churn.data_checks import DataQualityError, validate_and_clean


def test_clean_data_has_no_defects(clean_small):
    od = clean_small["orders"]
    assert od.invoice_id.is_unique
    assert (od.gross_value_idr > 0).all()
    assert od.order_date.max() <= pd.Timestamp("2026-09-30")
    assert clean_small["outlets"].city.notna().all()


def test_fix_rules_are_logged(raw_small):
    _, log = validate_and_clean(raw_small)
    fixed = {r.rule: r.n_rows for r in log if r.severity == "FIX"}
    assert fixed["no exact duplicate rows"] > 0          # generator injects duplicates
    assert fixed["gross_value_idr > 0"] > 0             # and negative values


def test_orphan_orders_block_the_pipeline(raw_small):
    bad = {k: v.copy() for k, v in raw_small.items()}
    row = bad["orders"].iloc[[0]].assign(outlet_id="OUT-99999", invoice_id="INV-X")
    bad["orders"] = pd.concat([bad["orders"], row])
    with pytest.raises(DataQualityError, match="outlet master"):
        validate_and_clean(bad)


def test_missing_column_blocks_the_pipeline(raw_small):
    bad = {k: v.copy() for k, v in raw_small.items()}
    bad["orders"] = bad["orders"].drop(columns="days_paid_late")
    with pytest.raises(Exception):
        validate_and_clean(bad)
