# AI-Use Log

Kept for the Week 11 AI-use verification. The rule I followed: **AI could write code and challenge my thinking; every decision about the business problem, metric, threshold and recommendation is mine and is defended in the reports.**

| Week | Tool | What I used it for | What I kept / changed / rejected | Where to verify |
|---|---|---|---|---|
| 1 | Gemini | Critique of my churn definition (30 vs 60 vs 90 days) | Kept 60 days; Gemini favoured 90. Rejected after the Head of Sales said 90 days is "too late, they've already switched". | `01_decision_brief.md` §1 |
| 2 | Gemini | "What statistical test fits a binary outcome vs a skewed continuous feature?" | Used Mann–Whitney for SKU change; I added the independence caveat myself (single snapshot) | `stats_evidence.py` |
| 3 | none | p-chart built by hand in Sheets first, then reproduced in Python | — | `figures/p_chart_region.png` |
| 4 | Claude Code | Refactor Colab notebook into `src/segar_churn/` modules; write `Makefile` | Accepted structure. Rewrote `features.py` window logic myself after spotting an off-by-one (used `>=` on the snapshot start, double-counting one day) | `features.py::_window` |
| 4 | Claude Code | Generate the no-leakage test | Accepted; I added the label-definition test | `tests/test_features.py` |
| 5 | Gemini | Sceptical review of the model set-up (5 challenges) | 3 accepted, 1 partly, 1 rejected (SMOTE). Full table in the model report | `02_model_report.md` §4 |
| 5 | Mentor | Live check-in | Two follow-ups carried forward (hit-rate guardrail, cold-run) | `02_model_report.md` §6 |
| 6 | Claude Code | Reason-code function (occlusion) | First version showed "Order frequency +1900%" as a risk reason. I specified the direction-aware filter and the actionable-features-only rule | `explain.py::RISK_DIRECTION` |
| 7 | Claude Code | GitHub Actions CI, README draft | Accepted CI as-is; rewrote README "Results" in my own words | `.github/workflows/ci.yml` |
| 8 | Claude | Generated first HTML dashboard from my spec and `dashboard_data.json` | I specified the four tabs, KPI choice and the warung lane label; removed a "model confidence" gauge it added (not meaningful to reps) | `app/dashboard_template.html` |
| 9 | Gemini | Stakeholder pushback role-play (4 personas) | 3 changes made, 1 declined with reasoning | `06_stakeholder_pushback_revision_note.md` |
| 10 | Gemini | Red-team: "How would this product fail in production?" | Added decommission criterion and rep-skip-rate check to governance plan | `05_monitoring_governance_plan.md` |

### Things I can explain without notes (panel prep)

- Why ranking by probability alone loses to the baseline on revenue (35.5% vs 43.9%), and why expected loss fixes it.
- Why the split is temporal, and what the leakage test does line by line.
- Why logistic regression was chosen although gradient boosting scored higher on test.
- How the 5% floor was chosen (on validation, with a pre-stated 3-point give-up rule) and why that is not tuning on test.
- What the holdout can and cannot prove.
