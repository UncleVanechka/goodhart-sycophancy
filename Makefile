PY ?= .venv/bin/python
CFG ?= configs

# Reproduce the reported tables and figures: make data baseline bon grpo-all ci plots
# One GRPO run with another seed: make grpo SEED=43 (run name gets -s43)
SEED ?=
SEED_ARG = $(if $(SEED),--seed $(SEED))

.PHONY: setup setup-dev test lint data baseline bon grpo grpo-mitigation grpo-persona-mix grpo-all \
	grpo-persona-mix-600 grpo-floor reeval reeval-600 reeval-floor are-you-sure prereg-v2 \
	prereg-floor prereg-are-you-sure sample-rate rescore ci plots clean

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt -e .

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

# 600-step persona mixture (docs/PREREG.md v2), not part of grpo-all; its runs go to
# results/grpo_600/: make grpo-persona-mix-600 SEED=44, then SEED=45, 46
grpo-persona-mix-600:
	$(PY) -m goodhart.experiments.train_grpo --config $(CFG)/persona_mix_600.yaml $(SEED_ARG)

# Reply-floor arm (docs/PREREG.md v3), not part of grpo-all: make grpo-floor, then SEED=43, 44
grpo-floor:
	$(PY) -m goodhart.experiments.train_grpo --config $(CFG)/grpo_floor.yaml $(SEED_ARG)

# Analysis of saved adapters (GPU; the policies and inputs come from the config):
# greedy re-evaluation with per-item outputs, sampled replies rated per sample, and
# saved replies re-rated by other judges or personas
reeval:
	$(PY) -m goodhart.experiments.reeval --config $(CFG)/reeval.yaml

# Checkpoints 100-600 of the 600-step persona-mixture runs (docs/PREREG.md v2)
reeval-600:
	$(PY) -m goodhart.experiments.reeval --config $(CFG)/reeval_600.yaml

reeval-floor:
	$(PY) -m goodhart.experiments.reeval --config $(CFG)/reeval_floor.yaml

# Two-turn "are you sure?" evaluation of the saved adapters (docs/PREREG.md v3; GPU, no judge)
are-you-sure:
	$(PY) -m goodhart.experiments.are_you_sure --config $(CFG)/are_you_sure.yaml

# Registered reading of the three 600-step persona-mixture runs of docs/PREREG.md v2 (CPU):
# endpoint and class of every checkpoint, the rule and the predictions ->
# results/analysis/prereg_v2.{json,md}. It reads the stored runs (results/grpo_600) and the
# per-question rows, which are not committed: the checkpoints of make reeval-600 and the base
# policy of make reeval (the reference of the paired counts). ARGS adds arguments or replaces
# the directories (the module's --help), e.g.
# make prereg-v2 ARGS="--reeval <dir> <dir> --base-items <file>"
prereg-v2:
	$(PY) -m goodhart.analysis.prereg_v2 --runs results/grpo_600 \
		--reeval results/reeval_600 results/reeval --base-items results/reeval/base/items.jsonl $(ARGS)

# Registered reading of the two experiments of docs/PREREG.md v3 (CPU): labels, classes, gates
# and predictions -> results/analysis/prereg_floor.{json,md}, prereg_are_you_sure.{json,md}.
# ARGS adds arguments or replaces the directories (each module's --help), e.g.
# make prereg-floor ARGS="--runs <dir> <dir> <dir> --reeval <dir>"
prereg-floor:
	$(PY) -m goodhart.analysis.floor_arm --runs results/grpo_floor --reeval results/reeval_floor $(ARGS)

prereg-are-you-sure:
	$(PY) -m goodhart.analysis.are_you_sure --stage-dir results/are_you_sure $(ARGS)

sample-rate:
	$(PY) -m goodhart.experiments.sample_rate --config $(CFG)/sample_rate.yaml

rescore:
	$(PY) -m goodhart.experiments.rescore --config $(CFG)/rescore.yaml

ci:
	$(PY) -m goodhart.experiments.confidence

plots:
	$(PY) -m goodhart.experiments.plots --results results --out figures

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
