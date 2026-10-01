"""Rebuilds notebooks/walkthrough.ipynb (run after `make all`)."""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Churn Early-Warning: walkthrough\n\nA guided tour of the capstone for reviewers. Every cell calls the tested package in `src/segar_churn`; nothing here is logic the repo doesn't already test.\n\n**In Colab:** run the first cell to clone the repo and build everything (~1 min)."),
    code("import os, sys, json\nif 'google.colab' in sys.modules and not os.path.exists('segar-churn-early-warning'):\n    !git clone <your-repo-url> segar-churn-early-warning\n    %cd segar-churn-early-warning\n    !pip -q install -r requirements.txt\n    !PYTHONPATH=src python -m segar_churn.pipeline\nROOT = os.path.abspath('..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()\nsys.path.insert(0, os.path.join(ROOT, 'src'))\nimport pandas as pd\nfrom IPython.display import Image, Markdown, display\nfrom segar_churn import config as C\nprint('repo root:', ROOT)"),
    md("## 1. Is the data trustworthy? (Checkpoint 1)"),
    code("display(Markdown((C.REPORTS / 'data_quality_log.md').read_text()))"),
    md("## 2. Is the problem real, and is it changing?"),
    code("ev = json.loads((C.REPORTS / 'evidence.json').read_text())\nr = ev['churn_rate']\nprint(f\"60-day churn: {r['rate']:.1%} (95% CI {r['ci95'][0]:.1%}–{r['ci95'][1]:.1%}), n={r['n']}\")\nfor h in ['H1_late_payers', 'H2_sku_shrink', 'H3_no_visit', 'H4_region']:\n    print(f\"{h}: p = {ev[h]['p_value']:.1e}  ({ev[h]['question']})\")\ndisplay(Image(C.FIGURES / 'p_chart_region.png'))"),
    md("## 3. Does the model beat what the team does today? (Checkpoint 2A)"),
    code("m = json.loads((C.REPORTS / 'metrics.json').read_text())\ncols = ['revenue_captured_at_k', 'precision_at_k', 'churners_caught_at_k', 'roc_auc', 'pr_auc']\ndisplay(pd.DataFrame(m['test']).T[cols].round(3))\ndisplay(Image(C.FIGURES / 'gains_revenue_captured.png'))"),
    md("Ranking by probability only (`*_prob_only`) **loses** to the recency baseline on revenue. Ranking by expected revenue lost is what makes the product work."),
    md("## 4. The deployed policy after stakeholder pushback (Checkpoint 3)"),
    code("display(pd.DataFrame(m['test_policy_comparison']).T.round(3))"),
    md("## 5. Who does the list reach? (bias review, Checkpoint 2B)"),
    code("display(Image(C.FIGURES / 'bias_review.png'))"),
    md("## 6. This month's visit list"),
    code("vl = pd.read_csv(C.OUTPUTS / 'visit_list_2026-09-30.csv')\ndisplay(vl[['priority_rank','lane','outlet_name','outlet_type','region','churn_probability','revenue_at_stake_m','expected_loss_m','reason_1','action']].head(15))\nprint(vl.groupby('region').size().sort_values(ascending=False))"),
    md("## 7. Drift check\n\nPSI < 0.10 stable · 0.10–0.25 watch · > 0.25 investigate."),
    code("display(pd.read_csv(C.OUTPUTS / 'drift_report_2026-09-30.csv'))"),
    md("Open `app/dashboard.html` for the stakeholder view."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
here = Path(__file__).parent
NotebookClient(nb, timeout=300, resources={"metadata": {"path": str(here)}}).execute()
nbf.write(nb, here / "walkthrough.ipynb")
print("wrote notebooks/walkthrough.ipynb")
