"""The GPU-time summary of the runner jobs, on synthetic summaries; and the committed outputs
reproduced from the fetched runner outputs when they are on disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from goodhart.analysis import compute
from goodhart.io import write_json


def _job(root: Path, name: str, stage: str, seconds: int, rc: int = 0) -> None:
    write_json(root / name / "summary.json", {"stage": stage, "rc": rc, "seconds": seconds})


def test_totals_by_stage_and_failed_jobs(tmp_path):
    _job(tmp_path, "a", "grpo", 3600)
    _job(tmp_path, "b", "grpo", 1800, rc=1)
    _job(tmp_path, "c", "reeval", 900)
    payload = compute.analyze(tmp_path)
    assert payload["jobs"] == 3
    assert payload["seconds"] == 6300
    assert payload["hours"] == 1.75
    assert payload["failed"] == [{"stage": "grpo", "rc": 1, "seconds": 1800}]
    assert payload["stages"]["grpo"] == {"jobs": 2, "seconds": 5400, "hours": 1.5}
    report = compute.render_markdown(payload)
    assert "6300 s = 1.75 GPU-hours" in report
    assert "| reeval | 1 | 900 | 0.25 |" in report


def test_no_summaries_and_missing_seconds(tmp_path):
    with pytest.raises(FileNotFoundError):
        compute.analyze(tmp_path)
    write_json(tmp_path / "x" / "summary.json", {"stage": "grpo", "rc": 0})
    with pytest.raises(ValueError):
        compute.analyze(tmp_path)


def test_main_writes_json_and_md(tmp_path):
    runner = tmp_path / "fetched"
    _job(runner, "a", "bon", 7200)
    out = tmp_path / "compute.json"
    compute.main(["--runner-results", str(runner), "--out", str(out)])
    assert out.exists() and out.with_suffix(".md").exists()


def test_committed_outputs_reproduce(tmp_path):
    root = Path(__file__).resolve().parents[1]
    runner = root / "results" / "fetched"
    if not list(runner.glob("*/summary.json")):
        pytest.skip("fetched runner outputs not on disk")
    out = tmp_path / "compute.json"
    compute.main(["--runner-results", str(runner), "--out", str(out)])
    committed = root / "results" / "analysis" / "compute.json"
    assert out.read_bytes() == committed.read_bytes()
    assert out.with_suffix(".md").read_bytes() == committed.with_suffix(".md").read_bytes()
