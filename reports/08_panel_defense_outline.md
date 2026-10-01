# Week 11 Panel Defense — 20-minute run sheet

| Time | Segment | What I show | Key line |
|---|---|---|---|
| 0:00–4:00 | **Problem & data** | Decision brief one-pager; p-chart | "Every month about Rp 10 bn of annual revenue walks out, and our follow-up list finds less than half of it." |
| 4:00–10:00 | **Statistics & model, live** | `make test` passing; gains curve; ranking-by-P vs P×revenue table; leakage test code | "Ranking by probability alone loses to what reps do today. Ranking by expected loss captures 69% vs 44%." |
| 10:00–14:00 | **Product & recommendation** | Dashboard: visit list → click an outlet → reasons & action; Jawa Timur alert; memo ask | "Approve a 3-month pilot with a 10% holdout, so in January we know whether visits save outlets, not just whether the model predicts churn." |
| 14:00–20:00 | **Q&A + AI-use verification** | AI-use log; be ready to modify a threshold live | — |

## Questions I expect, and short answers

1. **"Isn't AUC 0.89 suspicious?"** Recency alone gets 0.83, so the base signal is strong. Leakage is ruled out by a test that deletes all future data and checks features are unchanged.
2. **"Why not gradient boosting? It scored higher."** It was not better on validation, where I made the choice. +2.4 points on test is within month-to-month noise. It is the registered challenger for the first quarterly review.
3. **"You are ignoring small shops."** I was, in v1.0: 0% of churning warungs reached. v1.2 reserves 40 slots and reaches 34%. The trade-off is on the dashboard, not hidden.
4. **"How do you know visits work?"** I don't yet. That is what the holdout is for. The memo's value estimate is labelled as an assumption (1-in-4 save rate) with a 1-in-10 sensitivity.
5. **"What if the data changes?"** Monthly PSI and outcome back-test; red thresholds roll back to the recency list automatically.
6. **"Live: change capacity to 150 visits."** `VISIT_CAPACITY = 150` in `config.py` → `make train score dashboard` (≈ 20 s).
