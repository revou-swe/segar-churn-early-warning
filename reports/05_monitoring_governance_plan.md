# Monitoring & Governance Plan

Sized to the product's actual risk. Reproduce the monthly checks with `make score` → `outputs/drift_report_<date>.csv`.

---

## 1. Risk sizing

| Question | Answer | Implication |
|---|---|---|
| Does the model take an action by itself? | No. It produces a ranked list; a manager assigns visits and a rep decides. | Human in the loop at two levels |
| Can a wrong prediction harm a partner? | Low. A false alarm means an extra friendly visit. A miss means status quo. | No customer-facing consequence (no pricing, credit or termination decisions) |
| Personal data? | None. Outlet IDs, transactions and rep IDs only; owner names and phones excluded. | No UU PDP (Law 27/2022) processing of personal data in the model |
| Could it systematically disadvantage a group? | Yes: revenue weighting ignores small outlets. | Reserved warung lane + monthly coverage report |
| Cost of silent failure | Medium: reps chase the wrong outlets and lose trust. | Monthly drift and outcome checks |

**Overall: low–medium risk.** Governance is deliberately light: one owner, one monthly checklist, clear triggers. No model-risk committee.

## 2. Roles

| Role | Who | Responsibility |
|---|---|---|
| Model owner | Sales Ops analyst | runs the monthly job, checks this list, retrains |
| Business owner | Head of Sales | approves the list policy (capacity, lanes, floor) and any change to it |
| Reviewer | Finance BI lead | quarterly review of performance and coverage |
| Users | Area Sales Managers, reps | act on the list; log skips with a reason |

## 3. Monthly checklist (1st working day)

| # | Check | Source | Green | Amber → action | Red → action |
|---|---|---|---|---|---|
| 1 | Data checks pass | `reports/data_quality_log.md` | no BLOCK | FIX counts > 3× usual → ask ERP team | any BLOCK → **do not publish list** |
| 2 | Input drift (PSI per feature) | `outputs/drift_report_*.csv` | all < 0.10 | any 0.10–0.25 → note cause | any > 0.25 → investigate before publishing |
| 3 | Score drift | mean predicted P vs last 3 months | within ±20% | ±20–40% → check region mix | > ±40% → hold list, investigate |
| 4 | Realised performance (outcomes from 60 days ago) | back-test on the month that has matured | revenue captured ≥ 55% | 45–55% → watch | < 45% two months running (≈ today's method) → **retrain or roll back to recency list** |
| 5 | Coverage by segment | bias table | every type & region ≥ 20% of its churners reached | any < 20% → review lane sizes | any at 0% → fix before next run |
| 6 | Rep skip rate | CRM skip log | < 15% | 15–30% → read the reasons | > 30% → trust problem; review with ASMs |

**September 2026 run:** checks 1–3 green except **tenure PSI = 0.22 (amber)**. Cause understood: the partner base ages every month, so tenure drifts by construction. Action: none now; at the next retrain, bucket tenure (0–6, 6–24, 24+ months) so it stops drifting mechanically.

## 4. Measuring whether visits work (the holdout)

The model predicts churn; it does not prove visits prevent it (Checkpoint 1, H3). From October:

- Each month, **randomly hold out 10% of listed outlets** (≈ 20) from the extra visit. Stratify by lane.
- After 60 days, compare churn in visited vs held-out outlets. Pool across the 3-month pilot (≈ 60 held out vs ≈ 540 visited).
- Decision rule for January: scale up if the visited group's churn is lower with a one-sided 90% CI excluding zero; otherwise rethink the *intervention*, not the model.
- Ethics: held-out outlets get normal service (their usual route visits). Only the *extra* retention visit is withheld.

## 5. Retraining & change control

| Trigger | Action |
|---|---|
| Quarterly (Jan, Apr, Jul, Oct) | Retrain on all matured snapshots; compare against current model **and the gradient-boosting challenger** on the latest 3 months; promote only if revenue captured improves by ≥ 2 pts without coverage falling below thresholds |
| Red on check 4 | Retrain immediately; if no improvement, revert to recency list and notify Head of Sales |
| New region / outlet type / ERP change | Treat as new data source: rerun Checkpoint 1 checks before scoring |
| Policy change (capacity, lanes, floor) | Head of Sales approves; record in `CHANGELOG` with the back-test numbers |

Every model version is a git tag. `models/churn_model.joblib` stores the training window, and `reports/metrics.json` records the exact metrics it was promoted on.

## 6. Known limitations (stated to users)

- Revenue at stake is last 90 days × 4: seasonal outlets are mis-sized around Ramadan and year-end.
- The model learns from history; a sudden shock (a new competitor) shows up in the data before the model fully reflects it. The p-chart on the dashboard is the early alarm for that.
- Reasons are associations, not diagnoses. They are conversation starters for the rep.
- **Decommission criterion:** if after two quarters the holdout shows no retention effect *and* reps skip > 30%, retire the list and keep only the regional p-chart.
