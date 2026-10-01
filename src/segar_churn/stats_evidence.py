"""Checkpoint 1 statistical evidence: is the problem real, how big is it, and is it changing?

1. Size of the problem      -> churn rate with Wilson 95% CI, revenue lost per month
2. Hypothesis tests         -> do the signals we plan to use actually separate churners?
3. Statistical quality control (p-chart) -> is churn stable, or is something new happening?

Independence note: tests 1-2 use a single snapshot (one row per outlet) so observations
are independent. The p-chart uses every monthly snapshot by design.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from . import config as C
from .plotstyle import INK, MUTED, PALETTE, style

BASELINE_SNAPSHOTS = (6, 14)       # Apr-2025 .. Dec-2025 defines "normal" for the control chart


def main() -> None:
    style()
    panel = pd.read_csv(C.DATA_PROCESSED / "panel.csv")
    ev: dict = {}

    # ---------- 1. size of the problem ----------
    snap = panel[panel.snapshot_idx == C.TEST_LAST]            # latest labelled month, one row per outlet
    k, n = int(snap.churned.sum()), len(snap)
    lo, hi = proportion_confint(k, n, method="wilson")
    ev["latest_snapshot"] = snap.snapshot.iat[0]
    ev["churn_rate"] = {"k": k, "n": n, "rate": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    by_type = []
    for t, g in snap.groupby("outlet_type"):
        lo_, hi_ = proportion_confint(g.churned.sum(), len(g), method="wilson")
        by_type.append({"outlet_type": t, "n": len(g), "rate": round(g.churned.mean(), 4),
                        "ci_lo": round(lo_, 4), "ci_hi": round(hi_, 4),
                        "revenue_lost_idr_m": round(float((g.churned * g.revenue_at_stake_m).sum()), 1)})
    ev["churn_by_type"] = by_type
    monthly_loss = panel.assign(l=panel.churned * panel.revenue_at_stake_m).groupby("snapshot").l.sum()
    ev["annualised_revenue_lost_per_month_idr_m"] = {"mean": round(float(monthly_loss.mean()), 0),
                                                     "latest": round(float(monthly_loss.iloc[-1]), 0)}

    # ---------- 2. hypothesis tests ----------
    late = snap.avg_days_late_90d > 14
    cnt = np.array([snap.churned[late].sum(), snap.churned[~late].sum()])
    nobs = np.array([late.sum(), (~late).sum()])
    z, p = proportions_ztest(cnt, nobs)
    ev["H1_late_payers"] = {
        "question": "Do outlets paying >14 days late on average churn more?",
        "rate_late": round(cnt[0] / nobs[0], 4), "rate_on_time": round(cnt[1] / nobs[1], 4),
        "n_late": int(nobs[0]), "n_on_time": int(nobs[1]), "z": round(float(z), 2), "p_value": float(f"{p:.2e}")}

    ch, re = snap[snap.churned == 1].sku_trend, snap[snap.churned == 0].sku_trend
    u, p = stats.mannwhitneyu(ch, re, alternative="less")
    ev["H2_sku_shrink"] = {
        "question": "Is SKU-range change lower (shrinking) for outlets that go on to churn?",
        "median_churned": round(float(ch.median()), 3), "median_retained": round(float(re.median()), 3),
        "U": float(u), "p_value": float(f"{p:.2e}"), "effect_rank_biserial": round(1 - 2 * u / (len(ch) * len(re)), 3)}

    vis0 = snap.visits_60d == 0
    cnt = np.array([snap.churned[vis0].sum(), snap.churned[~vis0].sum()])
    nobs = np.array([vis0.sum(), (~vis0).sum()])
    z, p = proportions_ztest(cnt, nobs)
    ev["H3_no_visit"] = {
        "question": "Do outlets with zero rep visits in 60 days churn more?",
        "rate_no_visit": round(cnt[0] / nobs[0], 4), "rate_visited": round(cnt[1] / nobs[1], 4),
        "n_no_visit": int(nobs[0]), "z": round(float(z), 2), "p_value": float(f"{p:.2e}"),
        "caution": "Observational: reps may already avoid outlets they think are lost (reverse causality). "
                   "Motivates a randomised holdout in the governance plan."}

    tab = pd.crosstab(snap.region, snap.churned)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    ev["H4_region"] = {"question": "Does churn rate differ by region?", "chi2": round(float(chi2), 2), "dof": int(dof),
                       "p_value": float(f"{p:.2e}"),
                       "rates": (tab[1] / tab.sum(axis=1)).round(4).to_dict()}

    # ---------- 3. p-chart by region ----------
    reg = panel.groupby(["region", "snapshot_idx", "snapshot"]).churned.agg(["sum", "size"]).reset_index()
    regions = sorted(panel.region.unique())
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    ooc = {}
    for ax, r in zip(axs.flat, regions):
        g = reg[reg.region == r].sort_values("snapshot_idx")
        base = g[g.snapshot_idx.between(*BASELINE_SNAPSHOTS)]
        pbar = base["sum"].sum() / base["size"].sum()
        sig = np.sqrt(pbar * (1 - pbar) / g["size"])
        ucl, lcl = pbar + 3 * sig, np.maximum(0, pbar - 3 * sig)
        prop = g["sum"] / g["size"]
        x = pd.to_datetime(g.snapshot)
        out = prop > ucl
        ooc[r] = g.snapshot[out].tolist()
        ax.fill_between(x, lcl, ucl, color="#EEF4F1", step="mid")
        ax.plot(x, prop, color=PALETTE[0] if not out.any() else PALETTE[3], marker="o", ms=3.5, lw=1.5)
        ax.scatter(x[out], prop[out], color=PALETTE[3], zorder=5, s=36)
        ax.axhline(pbar, color=MUTED, lw=0.8, ls="--")
        ax.set_title(r + ("  ⚠ out of control" if out.any() else ""), loc="left", fontsize=10.5,
                     color=PALETTE[3] if out.any() else INK)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle("p-chart: 60-day churn rate by region (band = 3σ limits from Apr–Dec 2025)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(); fig.savefig(C.FIGURES / "p_chart_region.png", dpi=160); plt.close(fig)
    ev["p_chart_out_of_control"] = {r: v for r, v in ooc.items() if v}

    # churn by type with CI
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bt = pd.DataFrame(by_type).sort_values("rate")
    ax.errorbar(bt.rate * 100, bt.outlet_type, xerr=[(bt.rate - bt.ci_lo) * 100, (bt.ci_hi - bt.rate) * 100],
                fmt="o", color=PALETTE[0], ecolor=MUTED, capsize=4)
    ax.set_xlabel("60-day churn rate, % (95% Wilson CI)")
    ax.set_title(f"Churn by outlet type — snapshot {ev['latest_snapshot']}", loc="left")
    fig.tight_layout(); fig.savefig(C.FIGURES / "churn_by_type_ci.png", dpi=160); plt.close(fig)

    (C.REPORTS / "evidence.json").write_text(json.dumps(ev, indent=2, default=str))
    print(json.dumps(ev, indent=2, default=str))


if __name__ == "__main__":
    main()
