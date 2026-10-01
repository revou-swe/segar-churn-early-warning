"""Checkpoint 2B: explainability + bias review.

* Global: permutation importance (model-agnostic, measured on the TEST window)
          and standardised logistic coefficients (direction of effect).
* Local:  reason codes per outlet — see reason_codes(), reused by score.py.
* Bias:   does the deployed visit-list policy serve every region and outlet type,
          or does it systematically skip some partners?
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config as C
from .modeling import FEATURES, MODEL_FACTORIES, visit_list_mask
from .plotstyle import INK, MUTED, PALETTE, style
from .train import split


# Reason codes are restricted to things a sales rep can see and act on, and only count
# when the outlet is worse than a typical healthy outlet in the risky direction.
# (+1 = higher is riskier, -1 = lower is riskier.) Structural features such as outlet
# type or order size still drive the score but are never shown as a "reason".
RISK_DIRECTION = {
    "recency_days": +1, "orders_90d": -1, "order_trend": -1, "revenue_trend": -1,
    "sku_breadth_90d": -1, "sku_trend": -1, "avg_days_late_90d": +1, "share_late_14_90d": +1,
    "avg_discount_pct_90d": +1, "visits_60d": -1, "days_since_visit": +1, "complaints_90d": +1,
    "tenure_months": -1, "region": 0,
}


def reason_codes(model, X: pd.DataFrame, reference: pd.DataFrame, top: int = 3) -> list[list[tuple[str, float]]]:
    """Occlusion reason codes: for each actionable feature, swap the outlet's value for a
    'typical healthy outlet' value (median of retained outlets; most common region) and
    measure how much the churn probability drops. Larger drop = stronger reason.
    Model-agnostic: works for the logistic model and the gradient-boosting challenger."""
    base = model.predict_proba(X[FEATURES])[:, 1]
    healthy = reference[reference["churned"] == 0] if "churned" in reference else reference
    region_rate = reference.groupby("region")["churned"].mean() if "churned" in reference else None
    overall = reference["churned"].mean() if "churned" in reference else None
    contrib = {}
    for f, direction in RISK_DIRECTION.items():
        typical = healthy[f].median() if f in C.NUMERIC_FEATURES else healthy[f].mode().iat[0]
        Xr = X[FEATURES].copy()
        Xr[f] = typical
        delta = base - model.predict_proba(Xr)[:, 1]
        if direction:
            worse = (X[f].to_numpy() - typical) * direction > 0
        else:  # region: only a reason if that region's churn is clearly above average
            worse = X[f].map(region_rate).to_numpy() > 1.5 * overall
        contrib[f] = np.where(worse, delta, 0.0)
    M = pd.DataFrame(contrib)
    out = []
    for i in range(len(M)):
        row = M.iloc[i]
        row = row[row > 0.003].sort_values(ascending=False).head(top)
        out.append([(f, float(v)) for f, v in row.items()])
    return out


def main() -> None:
    style()
    panel = pd.read_csv(C.DATA_PROCESSED / "panel.csv")
    tr, va, te = split(panel)
    meta = json.loads((C.REPORTS / "metrics.json").read_text())
    name = meta["selected_model"]
    model = MODEL_FACTORIES[name]().fit(pd.concat([tr, va])[FEATURES], pd.concat([tr, va]).churned)

    # ---------- global importance ----------
    pi = permutation_importance(model, te[FEATURES], te.churned, scoring="average_precision",
                                n_repeats=5, random_state=C.SEED)
    imp = pd.DataFrame({"feature": FEATURES, "importance": pi.importances_mean, "sd": pi.importances_std}) \
        .sort_values("importance")
    imp["label"] = imp.feature.map(C.FEATURE_LABELS)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.barh(imp.label, imp.importance, xerr=imp.sd, color=PALETTE[0], ecolor=MUTED)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled (test window)")
    ax.set_title("What the model relies on", loc="left")
    fig.tight_layout(); fig.savefig(C.FIGURES / "permutation_importance.png", dpi=160); plt.close(fig)

    coef = None
    if name == "logistic_regression":
        names = model.named_steps["pre"].get_feature_names_out()
        coef = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
        coef.round(3).to_csv(C.REPORTS / "logistic_coefficients.csv", header=["coef_per_sd"])

    # ---------- bias review on the deployed policy ----------
    prob = model.predict_proba(te[FEATURES])[:, 1]
    rows = []
    floor = meta["probability_floor"]
    for policy, reserved, fl in [("v1.0 pure expected revenue", 0, 0.0),
                                 ("v1.2 deployed (Warung lane + floor)", C.RESERVED_SMALL_SLOTS, floor)]:
        sel = visit_list_mask(te, prob, reserved=reserved, floor=fl)
        d = te.assign(sel=sel)
        for dim in ["outlet_type", "region"]:
            for grp, g in d.groupby(dim):
                ch = g[g.churned == 1]
                rows.append({"policy": policy, "dimension": dim, "group": grp, "outlets": len(g),
                             "churn_rate": round(g.churned.mean(), 3),
                             "share_of_list": round(g.sel.sum() / d.sel.sum(), 3),
                             "churners_reached": round(ch.sel.mean(), 3) if len(ch) else np.nan,
                             "false_alarm_rate": round(g[g.churned == 0].sel.mean(), 3),
                             "revenue_at_risk_reached": round((ch.sel * ch.revenue_at_stake_m).sum()
                                                              / max(1e-9, ch.revenue_at_stake_m.sum()), 3)})
    bias = pd.DataFrame(rows)
    bias.to_csv(C.REPORTS / "bias_review_table.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dim in zip(axs, ["outlet_type", "region"]):
        b = bias[bias.dimension == dim].pivot(index="group", columns="policy", values="churners_reached")
        y = np.arange(len(b))
        ax.barh(y - 0.2, b.iloc[:, 0] * 100, 0.38, color=MUTED, label=b.columns[0])
        ax.barh(y + 0.2, b.iloc[:, 1] * 100, 0.38, color=PALETTE[0], label=b.columns[1])
        ax.set_yticks(y, b.index)
        ax.set_xlabel("% of that group's churners who made the visit list")
        ax.set_title(f"Who gets reached? by {dim.replace('_', ' ')}", loc="left")
    axs[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(C.FIGURES / "bias_review.png", dpi=160); plt.close(fig)

    print(imp.sort_values("importance", ascending=False)[["label", "importance"]].round(4).to_string(index=False))
    if coef is not None:
        print(coef.round(2).to_string())
    print(bias.to_string(index=False))


if __name__ == "__main__":
    main()
