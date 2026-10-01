# Applied Data Science with AI — Capstone: Churn Early-Warning (PT Segar Nusantara)
PY ?= python
export PYTHONPATH := src

.PHONY: all setup data checks features evidence train explain score dashboard test clean

all:        ## full pipeline, raw data -> dashboard
	$(PY) -m segar_churn.pipeline

setup:      ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:       ## (re)generate the synthetic anchor dataset into data/raw/
	$(PY) -m segar_churn.generate_data
checks:     ## validate + clean raw data, write reports/data_quality_log.md
	$(PY) -m segar_churn.data_checks
features:   ## point-in-time monthly snapshots -> data/processed/panel.csv
	$(PY) -m segar_churn.features
evidence:   ## Checkpoint 1 statistics (CI, hypothesis tests, p-chart)
	$(PY) -m segar_churn.stats_evidence
train:      ## baselines vs models, temporal validation, model selection
	$(PY) -m segar_churn.train
explain:    ## permutation importance + bias review
	$(PY) -m segar_churn.explain
score:      ## score current month, write visit list + drift report
	$(PY) -m segar_churn.score
dashboard:  ## embed latest scores into app/dashboard.html
	$(PY) app/build_dashboard.py

test:       ## unit + data tests
	$(PY) -m pytest -q

clean:
	rm -rf data/raw/*.csv data/processed/*.csv models/*.joblib app/dashboard_data.json .pytest_cache
