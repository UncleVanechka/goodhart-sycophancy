"""Plots on synthetic results: PNGs are written, loading is shape-tolerant."""

import json
from pathlib import Path

import pytest

from goodhart.experiments import plots

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
N_VALUES = [1, 4, 16, 64]
ALPHAS = [0.0, 0.3]


def _metrics(gold: float, proxy: float, syc: float) -> dict:
    out = {
        "gold_accuracy": gold,
        "sycophancy_rate": syc,
        "flip_rate": syc * 0.8,
        "proxy_mean": proxy,
        "abstain_rate": 0.02,
        "n": 300,
    }
    for i, op in enumerate(("none", "correct", "wrong")):
        out[f"gold_accuracy@{op}"] = max(0.0, min(1.0, gold + 0.05 * (1 - i)))
        out[f"sycophancy_rate@{op}"] = syc if op == "wrong" else 0.0
        out[f"n@{op}"] = 100
    return out


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _bon_payload(model: str) -> dict:
    curves = {}
    for alpha in ALPHAS:
        curve = {}
        for k, n in enumerate(N_VALUES):
            curve[str(n)] = _metrics(
                gold=0.7 - 0.05 * k * (1 - alpha),
                proxy=6.0 + 0.8 * k,
                syc=0.2 + 0.1 * k * (1 - alpha),
            )
        curves[f"alpha={alpha}"] = curve
    return {
        "config": {"seed": 42},
        "git_commit": "deadbeef",
        "model": model,
        "n_values": N_VALUES,
        "curves": curves,
        "samples_file": f"results/bon/{model.lower()}_samples.jsonl",
    }


def _grpo_rows(nested: bool) -> list[dict]:
    rows = []
    for k, step in enumerate(range(0, 301, 50)):
        base = _metrics(gold=0.7 - 0.03 * k, proxy=6.0 + 0.4 * k, syc=0.2 + 0.05 * k)
        if nested:
            rows.append(
                {
                    "step": step,
                    "metrics": base,
                    "sycophancy_eval": {"sycophancy_rate": 0.3 + 0.04 * k, "n": 300},
                }
            )
        else:
            rows.append({"step": step, **base, "heldout_sycophancy_rate": 0.3 + 0.04 * k})
    return rows


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    for model, slug in (("Qwen/Qwen3-1.7B", "qwen3-1.7b"), ("Qwen/Qwen3-4B", "qwen3-4b")):
        _write_json(
            root / "baseline" / f"{slug}.json",
            {
                "config": {"seed": 42},
                "git_commit": None,
                "model": model,
                "metrics": _metrics(gold=0.65, proxy=5.5, syc=0.25),
            },
        )
        _write_json(root / "bon" / f"{slug}.json", _bon_payload(model))
        (root / "bon" / f"{slug}_samples.jsonl").write_text(
            '{"record_id": "x", "sample_idx": 0, "text": "Answer: A"}\n', encoding="utf-8"
        )
    for run, nested in (("grpo-immediate", False), ("grpo-delayed-a0.3", True)):
        run_dir = root / "grpo" / run
        run_dir.mkdir(parents=True)
        rows = _grpo_rows(nested)
        rows.reverse()
        with open(run_dir / "curve.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    _write_json(
        root / "grpo" / "grpo-immediate" / "summary.json",
        {"alpha": 0.0, "config": {"reward": {"alpha": 0.0}}, "model": "Qwen/Qwen3-1.7B"},
    )
    return root


def _assert_png(path: Path | None) -> None:
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == PNG_MAGIC


def test_load_results_shapes(results_dir: Path) -> None:
    results = plots.load_results(results_dir)
    assert set(results) == {"baseline", "bon", "grpo"}
    assert [p["model"] for p in results["baseline"]] == ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
    assert len(results["bon"]) == 2
    assert all("curves" in p for p in results["bon"])
    assert set(results["grpo"]) == {"grpo-immediate", "grpo-delayed-a0.3"}
    steps = [r["step"] for r in results["grpo"]["grpo-immediate"]]
    assert steps == sorted(steps)
    assert plots._alpha_of_run("grpo-immediate", results["grpo"]["grpo-immediate"]) == 0.0
    assert plots._alpha_of_run("grpo-delayed-a0.3", results["grpo"]["grpo-delayed-a0.3"]) == 0.3


def test_summary_alpha_falls_back_to_config_reward(tmp_path: Path) -> None:
    assert plots._summary_alpha(tmp_path / "missing.json") is None
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"config": {"reward": {"alpha": 0.6}}}), encoding="utf-8")
    assert plots._summary_alpha(path) == 0.6
    path.write_text(json.dumps({"config": {}}), encoding="utf-8")
    assert plots._summary_alpha(path) is None


def test_load_results_missing_dir(tmp_path: Path) -> None:
    results = plots.load_results(tmp_path / "nope")
    assert results == {"baseline": [], "bon": [], "grpo": {}}


def test_metric_lookup_flat_and_nested() -> None:
    assert plots._metric({"gold_accuracy": 0.5}, "gold_accuracy") == 0.5
    assert plots._metric({"metrics": {"gold_accuracy": 0.4}}, "gold_accuracy") == 0.4
    assert plots._metric({"metrics": {}}, "gold_accuracy") is None
    assert plots._heldout_sycophancy({"sycophancy_eval": {"sycophancy_rate": 0.3}}) == 0.3
    assert plots._heldout_sycophancy({"heldout_sycophancy_rate": 0.2}) == 0.2
    assert plots._heldout_sycophancy({"step": 1}) is None


def test_alpha_parsing() -> None:
    assert plots._alpha_of_key("alpha=0.3") == 0.3
    assert plots._alpha_of_key("0.6") == 0.6
    assert plots._alpha_of_key("weird") is None
    assert plots._alpha_of_run("grpo-delayed-a0.3", [{"step": 0}]) == 0.3
    assert plots._alpha_of_run("grpo-immediate", [{"step": 0}]) is None
    assert plots._alpha_of_run("x", [{"config": {"reward": {"alpha": 0.6}}}]) == 0.6


def test_each_plot_writes_png(results_dir: Path, tmp_path: Path) -> None:
    results = plots.load_results(results_dir)
    out = tmp_path / "figures"
    _assert_png(plots.plot_bon_curves(results["bon"], out / "bon_curves.png"))
    _assert_png(
        plots.plot_goodhart_frontier(results["bon"], results["grpo"], out / "goodhart_frontier.png")
    )
    _assert_png(plots.plot_grpo_curves(results["grpo"], out / "grpo_curves.png"))
    _assert_png(plots.plot_baseline(results["baseline"], out / "baseline.png"))


def test_frontier_from_grpo_only(results_dir: Path, tmp_path: Path) -> None:
    results = plots.load_results(results_dir)
    _assert_png(plots.plot_goodhart_frontier([], results["grpo"], tmp_path / "f.png"))


def test_plots_return_none_on_empty(tmp_path: Path) -> None:
    assert plots.plot_bon_curves([], tmp_path / "a.png") is None
    assert plots.plot_goodhart_frontier([], {}, tmp_path / "b.png") is None
    assert plots.plot_grpo_curves({}, tmp_path / "c.png") is None
    assert plots.plot_baseline([], tmp_path / "d.png") is None
    assert not list(tmp_path.glob("*.png"))


def test_main_writes_all_figures(results_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "figures"
    written = plots.main(["--results", str(results_dir), "--out", str(out)])
    names = sorted(p.name for p in written)
    assert names == ["baseline.png", "bon_curves.png", "goodhart_frontier.png", "grpo_curves.png"]
    for path in written:
        _assert_png(path)


def test_main_on_empty_results(tmp_path: Path) -> None:
    out = tmp_path / "figures"
    assert plots.main(["--results", str(tmp_path / "results"), "--out", str(out)]) == []
    assert not out.exists() or not list(out.glob("*.png"))


def test_training_reward_prefers_reward_mean_over_proxy() -> None:
    assert plots._training_reward({"reward_mean": 5.0, "proxy_mean": 9.0}) == 5.0
    assert plots._training_reward({"proxy_mean": 9.0}) == 9.0
    assert plots._training_reward({"gold_accuracy": 0.5}) is None


def test_frontier_skips_runs_without_proxy(tmp_path: Path) -> None:
    """A run that recorded only the training reward has no proxy axis and is left out."""
    immediate = [{"step": s, "proxy_mean": 7.0 + s / 100, "gold_accuracy": 0.6} for s in (0, 50)]
    delayed = [{"step": s, "reward_mean": 7.0 + s / 100, "gold_accuracy": 0.7} for s in (0, 50)]
    out = plots.plot_goodhart_frontier(
        [], {"grpo-immediate": immediate, "grpo-delayed-a0.3": delayed}, tmp_path / "f.png"
    )
    _assert_png(out)
    curves = plots.plot_grpo_curves(
        {"grpo-immediate": immediate, "grpo-delayed-a0.3": delayed}, tmp_path / "c.png"
    )
    _assert_png(curves)


def test_runs_with_the_same_alpha_are_aggregated(tmp_path: Path) -> None:
    def run(offset: float) -> list[dict]:
        return [
            {
                "step": s,
                "alpha": 0.3,
                "proxy_mean": 7.0 + s / 100 + offset,
                "gold_accuracy": 0.6 - offset,
            }
            for s in (0, 50, 100)
        ]

    grpo = {"grpo-delayed-s43-a0.3": run(0.0), "grpo-delayed-s44-a0.3": run(0.05)}
    groups = plots._group_runs(grpo)
    assert list(groups) == [0.3] and len(groups[0.3]) == 2
    agg = plots._aggregate(groups[0.3], lambda r: plots._metric(r, "gold_accuracy"))
    assert [a[0] for a in agg] == [0, 50, 100]
    assert (agg[0][1], agg[0][2], agg[0][3]) == pytest.approx((0.575, 0.55, 0.6))
    assert agg[0][4] == 2
    _assert_png(plots.plot_goodhart_frontier([], grpo, tmp_path / "f.png"))
    _assert_png(plots.plot_grpo_curves(grpo, tmp_path / "c.png"))
