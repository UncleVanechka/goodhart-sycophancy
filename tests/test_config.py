"""Config loading: base merge, env expansion, seeding."""

import random

import numpy as np

from goodhart.config import deep_merge, expand_env, load_config, set_seed


def _write(path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_deep_merge_recurses_and_overrides():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "list": [1]}
    override = {"a": 2, "nested": {"y": 3, "z": 4}, "list": [2, 3]}
    merged = deep_merge(base, override)
    assert merged == {"a": 2, "nested": {"x": 1, "y": 3, "z": 4}, "list": [2, 3]}
    assert base["nested"] == {"x": 1, "y": 2}


def test_load_config_merges_base_under_stage(tmp_path):
    _write(tmp_path / "base.yaml", "seed: 42\npaths:\n  data: data\n  results: results\n")
    _write(tmp_path / "stage.yaml", "seed: 7\npaths:\n  results: out\nmodel: m\n")
    cfg = load_config(tmp_path / "stage.yaml", base=tmp_path / "base.yaml")
    assert cfg == {"seed": 7, "paths": {"data": "data", "results": "out"}, "model": "m"}


def test_load_config_finds_base_next_to_stage(tmp_path, monkeypatch):
    _write(tmp_path / "base.yaml", "seed: 1\n")
    _write(tmp_path / "stage.yaml", "name: x\n")
    monkeypatch.chdir(tmp_path.parent)
    cfg = load_config(tmp_path / "stage.yaml", base="configs/base.yaml")
    assert cfg == {"seed": 1, "name": "x"}


def test_load_config_without_base(tmp_path):
    _write(tmp_path / "stage.yaml", "name: x\n")
    assert load_config(tmp_path / "stage.yaml", base=None) == {"name": "x"}
    assert load_config(tmp_path / "stage.yaml", base=tmp_path / "missing.yaml") == {"name": "x"}


def test_env_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("GH_TEST_URL", "http://host:8000/v1")
    monkeypatch.delenv("GH_TEST_MISSING", raising=False)
    _write(
        tmp_path / "stage.yaml",
        "vllm:\n  base_url: ${GH_TEST_URL}\n  token: ${GH_TEST_MISSING}\n"
        "items: ['${GH_TEST_URL}/x', 3]\nplain: 'no vars'\n",
    )
    cfg = load_config(tmp_path / "stage.yaml", base=None)
    assert cfg["vllm"] == {"base_url": "http://host:8000/v1", "token": ""}
    assert cfg["items"] == ["http://host:8000/v1/x", 3]
    assert cfg["plain"] == "no vars"


def test_expand_env_leaves_non_strings():
    assert expand_env({"a": 1, "b": None, "c": [True]}) == {"a": 1, "b": None, "c": [True]}


def test_repo_configs_load():
    cfg = load_config("configs/data.yaml")
    assert cfg["seed"] == 42
    assert cfg["paths"]["results"] == "results"
    assert cfg["sources"][0]["name"] == "mmlu"


def test_set_seed_is_reproducible():
    set_seed(123)
    a = (random.random(), float(np.random.rand()))
    set_seed(123)
    b = (random.random(), float(np.random.rand()))
    assert a == b
