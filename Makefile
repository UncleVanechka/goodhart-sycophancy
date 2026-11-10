PY ?= .venv/bin/python
CFG ?= configs

# Reproduce the reported tables and figures: make data baseline bon grpo-all ci plots
# One GRPO run with another seed: make grpo SEED=43 (run name gets -s43)
SEED ?=
SEED_ARG = $(if $(SEED),--seed $(SEED))

.PHONY: setup setup-dev test lint data baseline bon grpo grpo-mitigation grpo-persona-mix grpo-all ci plots clean

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
	$(PY) -m goodhart.data.sycophancy_eval

baseline:
	$(PY) -m goodhart.experiments.baseline --config $(CFG)/baseline.yaml

bon:
	$(PY) -m goodhart.experiments.best_of_n --config $(CFG)/bon.yaml

grpo:
	$(PY) -m goodhart.experiments.train_grpo --config $(CFG)/grpo.yaml $(SEED_ARG)

grpo-mitigation:
	$(PY) -m goodhart.experiments.train_grpo --config $(CFG)/mitigation.yaml $(SEED_ARG)

grpo-persona-mix:
	$(PY) -m goodhart.experiments.train_grpo --config $(CFG)/persona_mix.yaml $(SEED_ARG)

# Every run under results/grpo/: immediate and delayed reward at seeds 42-44, the persona
# mixture at seeds 42-43 (11 of the 13 GRPO runs in the write-up; the first-pass α = 0.3 / 0.6
# pair in results/grpo_first_pass/ is not rerun). ~61 GPU-hours on one A100 80 GB
grpo-all:
	for s in 42 43 44; do $(MAKE) grpo SEED=$$s && $(MAKE) grpo-mitigation SEED=$$s || exit 1; done
	for s in 42 43; do $(MAKE) grpo-persona-mix SEED=$$s || exit 1; done

ci:
	$(PY) -m goodhart.experiments.confidence

plots:
	$(PY) -m goodhart.experiments.plots --results results --out figures

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
