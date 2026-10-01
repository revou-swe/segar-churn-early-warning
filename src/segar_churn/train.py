"""Train, benchmark against baselines, select, and save the model (Checkpoint 2A/2B).

Protocol (strictly temporal — no random shuffling of outlet-months):
  1. Fit each candidate on TRAIN snapshots.
  2. Select the candidate with the best revenue-at-risk captured @K on VALIDATION.
  3. Refit the winner on TRAIN+VALIDATION and report once on the untouched TEST window.
  4. Refit on every labelled snapshot and save for live scoring.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config as C
from .modeling import (BASELINES, FEATURES, MODEL_FACTORIES, full_metrics, gains_curve, mask_metrics,
                       priority_score, visit_list_mask)
from .plotstyle import INK, MUTED, PALETTE, style

TARGET = "churned"


def split(panel: pd.DataFrame):
    tr = panel[panel.snapshot_idx <= C.TRAIN_LAST]
    va = panel[(panel.snapshot_idx > C.TRAIN_LAST) & (panel.snapshot_idx <= C.VALID_LAST)]
    te = panel[(panel.snapshot_idx > C.VALID_LAST) & (panel.snapshot_idx <= C.TEST_LAST)]
    return tr, va, te


def _recency_mask(df: pd.DataFrame) -> np.ndarray:
    d = df.assign(pos=np.arange(len(df)))
    idx = d.groupby("snapshot", group_keys=False).apply(lambda g: g.nlargest(C.VISIT_CAPACITY, "recency_days")).pos
    m = np.zeros(len(df), dtype=bool)
    m[idx.to_numpy()] = True
    return m


def evaluate_all(fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[dict, dict]:
    results, scores = {}, {}
    for name, fn in BASELINES.items():
        s = fn(eval_df)
        results[name] = full_metrics(eval_df, s)
        scores[name] = s
    for name, factory in MODEL_FACTORIES.items():
        model = factory().fit(fit_df[FEATURES], fit_df[TARGET])
        prob = model.predict_proba(eval_df[FEATURES])[:, 1]
        rank = priority_score(prob, eval_df)
        results[name] = full_metrics(eval_df, rank, prob)
        results[name + "_prob_only"] = full_metrics(eval_df, prob, prob)
        scores[name] = rank
        scores[name + "__prob"] = prob
    return results, scores


def main() -> None:
    style()
    panel = pd.read_csv(C.DATA_PROCESSED / "panel.csv")
    tr, va, te = split(panel)

    val_results, _ = evaluate_all(tr, va)
    candidates = {k: v for k, v in val_results.items() if k in MODEL_FACTORIES}
    winner = max(candidates, key=lambda k: candidates[k]["revenue_captured_at_k"])

    trva = pd.concat([tr, va])
    test_results, test_scores = evaluate_all(trva, te)

    # ---- choose the probability floor on VALIDATION (model fitted on train only) ----
    m_tr = MODEL_FACTORIES[winner]().fit(tr[FEATURES], tr[TARGET])
    p_va = m_tr.predict_proba(va[FEATURES])[:, 1]
    floor_grid = {f: mask_metrics(va, visit_list_mask(va, p_va, floor=f)) for f in C.FLOOR_GRID}
    best_rev = max(v["revenue_captured_at_k"] for v in floor_grid.values())
    floor = max(f for f, v in floor_grid.items() if v["revenue_captured_at_k"] >= best_rev - C.FLOOR_MAX_REVENUE_GIVEUP)

    # ---- policy comparison on TEST ----
    p_win = test_scores[winner + "__prob"]
    policy = {
        "today_recency_baseline": mask_metrics(te, _recency_mask(te)),
        "v1.0_pure_expected_revenue": mask_metrics(te, visit_list_mask(te, p_win, reserved=0)),
        f"v1.1_plus_{C.RESERVED_SMALL_SLOTS}_warung_slots": mask_metrics(te, visit_list_mask(te, p_win)),
        f"v1.2_plus_floor_{floor:.2f}": mask_metrics(te, visit_list_mask(te, p_win, floor=floor)),
    }

    final = MODEL_FACTORIES[winner]().fit(panel[FEATURES], panel[TARGET])
    C.MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": final, "name": winner, "features": FEATURES,
                 "trained_on": f"snapshots {panel.snapshot.min()} .. {panel.snapshot.max()}",
                 "train_reference": panel[FEATURES].copy()}, C.MODELS / "churn_model.joblib")

    split_info = {n: {"snapshots": f"{d.snapshot.min()} .. {d.snapshot.max()}", "rows": int(len(d)),
                      "churn_rate": round(float(d[TARGET].mean()), 4)} for n, d in
                  [("train", tr), ("validation", va), ("test", te)]}
    metrics = {"selected_model": winner, "selection_metric": "revenue_captured_at_k (validation)",
               "k": C.VISIT_CAPACITY, "splits": split_info,
               "validation": val_results, "test": test_results, "test_policy_comparison": policy,
               "deployed_policy": f"v1.2_plus_floor_{floor:.2f}", "probability_floor": floor,
               "floor_selection_validation": {str(k): v for k, v in floor_grid.items()}}
    (C.REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------------- figures ----------------
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    names = {"baseline_recency": "Baseline: longest-silent first",
             "baseline_rule_30d_by_revenue": "Baseline: silent >30d, biggest first",
             "logistic_regression": "Logistic regression (P × revenue)",
             "gradient_boosting": "Gradient boosting (P × revenue)"}
    colors = {"baseline_recency": MUTED, "baseline_rule_30d_by_revenue": "#B8B2A6",
              "logistic_regression": PALETTE[0], "gradient_boosting": PALETTE[1]}

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for k in names:
        g = gains_curve(te, test_scores[k])
        ax.plot(g.k, g.captured * 100, label=names[k], color=colors[k], lw=2.4 if k == winner else 1.6,
                ls="--" if k.startswith("baseline") else "-")
    ax.axvline(C.VISIT_CAPACITY, color=INK, lw=0.8, ls=":")
    ax.text(C.VISIT_CAPACITY + 8, 4, f"capacity = {C.VISIT_CAPACITY} visits/mo", color=INK, fontsize=9)
    ax.set_xlabel("Outlets on the visit list (K)")
    ax.set_ylabel("% of revenue-at-risk captured")
    ax.set_title("Test window: how much of the revenue we were about to lose does the list find?", loc="left")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(C.FIGURES / "gains_revenue_captured.png", dpi=160); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for k in ["logistic_regression", "gradient_boosting"]:
        p = test_scores[k + "__prob"]
        fpr, tpr, _ = roc_curve(te[TARGET], p)
        axs[0].plot(fpr, tpr, color=colors[k], label=f"{names[k].split(' (')[0]} AUC {test_results[k]['roc_auc']:.2f}")
        pr, rc, _ = precision_recall_curve(te[TARGET], p)
        axs[1].plot(rc, pr, color=colors[k], label=f"PR-AUC {test_results[k]['pr_auc']:.2f}")
    fpr, tpr, _ = roc_curve(te[TARGET], test_scores["baseline_recency"])
    axs[0].plot(fpr, tpr, color=MUTED, ls="--", label=f"Recency baseline AUC {test_results['baseline_recency']['roc_auc']:.2f}")
    axs[0].plot([0, 1], [0, 1], color="#DDD", lw=0.8)
    axs[0].set_xlabel("False positive rate"); axs[0].set_ylabel("True positive rate"); axs[0].set_title("ROC (test)", loc="left")
    axs[1].axhline(te[TARGET].mean(), color="#DDD", lw=0.8)
    axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_title("Precision–recall (test)", loc="left")
    for a in axs:
        a.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(C.FIGURES / "roc_pr.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for k in ["logistic_regression", "gradient_boosting"]:
        fr, mp = calibration_curve(te[TARGET], test_scores[k + "__prob"], n_bins=10, strategy="quantile")
        ax.plot(mp, fr, marker="o", ms=4, color=colors[k], label=names[k].split(" (")[0])
    ax.plot([0, 0.6], [0, 0.6], color="#DDD", lw=0.8)
    ax.set_xlabel("Predicted churn probability"); ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration (test)", loc="left"); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(C.FIGURES / "calibration.png", dpi=160); plt.close(fig)

    print(f"selected on validation: {winner}")
    print(pd.DataFrame(test_results).T[["revenue_captured_at_k", "precision_at_k", "churners_caught_at_k",
                                        "roc_auc", "pr_auc"]].round(3).to_string())
    print(pd.DataFrame(policy).T.round(3).to_string())
    print("model -> models/churn_model.joblib ; metrics -> reports/metrics.json")


if __name__ == "__main__":
    main()
