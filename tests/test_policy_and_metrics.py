import pytest
import numpy as np
import pandas as pd

from segar_churn.modeling import topk_metrics, visit_list_mask
from segar_churn.score import psi


def _toy(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"snapshot": "2026-01-31", "churned": rng.integers(0, 2, n),
                         "revenue_at_stake_m": rng.uniform(1, 100, n),
                         "outlet_type": np.where(np.arange(n) % 3 == 0, "Warung", "Grosir")})


def test_revenue_captured_is_one_when_list_covers_everything():
    d = _toy()
    assert topk_metrics(d, d.churned.to_numpy() * 10.0, k=len(d))["revenue_captured_at_k"] == pytest.approx(1.0)


def test_perfect_ranking_beats_random():
    d = _toy()
    perfect = topk_metrics(d, d.churned * d.revenue_at_stake_m, k=10)["revenue_captured_at_k"]
    rand = topk_metrics(d, np.random.default_rng(1).random(len(d)), k=10)["revenue_captured_at_k"]
    assert perfect > rand


def test_visit_list_respects_capacity_and_warung_lane():
    d = _toy(60)
    prob = np.random.default_rng(2).random(len(d))
    m = visit_list_mask(d, prob, k=20, reserved=5)
    assert m.sum() == 20
    assert (d[m].outlet_type == "Warung").sum() >= 5


def test_floor_keeps_low_risk_out_when_possible():
    d = _toy(60)
    prob = np.r_[np.full(30, 0.01), np.full(30, 0.5)]
    m = visit_list_mask(d, prob, k=20, reserved=0, floor=0.05)
    assert (prob[m] >= 0.05).all()


def test_psi_zero_for_identical_distributions():
    x = pd.Series(np.random.default_rng(3).normal(size=1000))
    assert psi(x, x) < 1e-6
    assert psi(x, x + 2) > 0.25
