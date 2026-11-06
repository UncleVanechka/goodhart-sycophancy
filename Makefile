PY ?= .venv/bin/python
CFG ?= configs

.PHONY: setup setup-dev test lint data baseline bon grpo grpo-mitigation plots clean

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt -e .

setup-dev:
	python3.11 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements-dev.txt -e .

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

data:
	$(PY) -m goodhart.data.build --config $(CFG)/data.yaml

baseline:
	$(PY) -m goodhart.experiments.baseline --config $(CFG)/bon.yaml

bon:
	$(PY) -m goodhart.experiments.best_of_n --config $(CFG)/bon.yaml

grpo:
	$(PY) -m goodhart.experiments.train_grpo --config $(CFG)/grpo.yaml

grpo-mitigation:
	$(PY) -m goodhart.experiments.train_grpo --config $(CFG)/mitigation.yaml

plots:
	$(PY) -m goodhart.experiments.plots --results results --out figures

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
