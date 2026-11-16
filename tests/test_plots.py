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
    explicit_null = {
        "alpha": None,
        "reward_kind": None,
        "config": {"reward": {"alpha": 0.6, "kind": "delayed"}},
    }
    path.write_text(json.dumps(explicit_null), encoding="utf-8")
    assert plots._summary_meta(path) == (0.6, "delayed")


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


def test_persona_mix_runs_form_their_own_series(tmp_path: Path) -> None:
    rows = [
        {"step": s, "gold_accuracy": 0.6, "proxy_mean": 7.9, "sycophancy_rate": 0.5}
        for s in (0, 50)
    ]
    immediate = [dict(r, alpha=0.0, reward_kind="immediate") for r in rows]
    mixed = [dict(r, alpha=0.0, reward_kind="persona_mix") for r in rows]
    grpo = {
        "grpo-immediate": immediate,
        "grpo-persona-mix": mixed,
        "grpo-delayed-a0.6": [dict(r, alpha=0.6) for r in rows],
    }
    assert plots._group_of_run("grpo-immediate", immediate) == 0.0
    assert plots._group_of_run("grpo-persona-mix", mixed) == "persona_mix"
    assert plots._group_of_run("grpo-persona-mix", [{"step": 0}]) == "persona_mix"
    cfg_rows = [{"config": {"reward": {"kind": "persona_mix", "alpha": 0.0}}}]
    assert plots._group_of_run("x", cfg_rows) == "persona_mix"
    groups = plots._group_runs(grpo)
    assert list(groups) == [0.0, 0.6, "persona_mix"]
    assert list(groups["persona_mix"]) == ["grpo-persona-mix"]
    assert plots._group_label("persona_mix", 1) == "GRPO · persona mix"
    assert plots._group_label(0.3, 2) == "GRPO · α=0.3 (2 runs)"
    colors = plots._series_colors([{"curves": {"alpha=0.0": {}, "alpha=1.0": {}}}], grpo)
    assert list(colors) == [0.0, 0.6, 1.0, "persona_mix"]
    assert len(set(colors.values())) == 4
    _assert_png(plots.plot_goodhart_frontier([], grpo, tmp_path / "f.png"))
    _assert_png(plots.plot_grpo_curves(grpo, tmp_path / "c.png"))


def test_load_results_attaches_reward_kind(tmp_path: Path) -> None:
    run = tmp_path / "grpo" / "grpo-persona-mix"
    run.mkdir(parents=True)
    (run / "curve.jsonl").write_text(json.dumps({"step": 0, "gold_accuracy": 0.6}) + "\n")
    (run / "summary.json").write_text(json.dumps({"alpha": 0.0, "reward_kind": "persona_mix"}))
    rows = plots.load_results(tmp_path)["grpo"]["grpo-persona-mix"]
    assert rows[0]["alpha"] == 0.0 and rows[0]["reward_kind"] == "persona_mix"


def _write_run(root: Path, subdir: str, name: str, rows: list[dict], summary: dict) -> None:
    run = root / subdir / name
    run.mkdir(parents=True)
    (run / "curve.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "summary.json").write_text(json.dumps(summary))


def _steps(last: int, **extra) -> list[dict]:
    return [
        {"step": s, "proxy_mean": 7.8 + s / 1000, "gold_accuracy": 0.65, **extra}
        for s in range(0, last + 1, 100)
    ]


FLOOR = {"min_words": 30, "value": 1.0}
MIX = {"alpha": 0.0, "reward_kind": "persona_mix"}


def test_load_results_reads_the_600_step_and_floor_directories(tmp_path: Path) -> None:
    floor_summary = {
        "alpha": 0.0,
        "reward_kind": "immediate",
        "floor": FLOOR,
        "config": {"reward": {"kind": "immediate", "alpha": 0.0, "floor": FLOOR}},
    }
    _write_run(tmp_path, "grpo", "grpo-persona-mix", _steps(300), MIX)
    _write_run(tmp_path, "grpo_600", "grpo-persona-mix-s44", _steps(600), MIX)
    _write_run(tmp_path, "grpo_600", "grpo-persona-mix", _steps(600), MIX)
    _write_run(tmp_path, "grpo_floor", "grpo-immediate-floor30", _steps(300), floor_summary)
    _write_run(tmp_path, "grpo_first_pass", "grpo-delayed-a0.3", _steps(300), {"alpha": 0.3})
    grpo = plots.load_results(tmp_path)["grpo"]
    assert list(grpo) == [
        "grpo-persona-mix",
        "grpo_600/grpo-persona-mix",  # a name taken by an earlier directory gets its own
        "grpo-persona-mix-s44",
        "grpo-immediate-floor30",
    ]
    assert grpo["grpo-immediate-floor30"][0]["reward_floor"] == FLOOR
    assert "reward_floor" not in grpo["grpo-persona-mix"][0]
    assert [r["step"] for r in grpo["grpo-persona-mix-s44"]][-1] == 600


def test_summary_fields_find_the_floor_at_top_level_or_in_config(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"alpha": 0.0, "floor": FLOOR}))
    assert plots._summary_fields(path) == {"alpha": 0.0, "reward_floor": FLOOR}
    path.write_text(json.dumps({"config": {"reward": {"kind": "immediate", "floor": FLOOR}}}))
    assert plots._summary_fields(path) == {"reward_kind": "immediate", "reward_floor": FLOOR}
    path.write_text(json.dumps({"alpha": 0.0, "floor": None, "config": {"reward": {}}}))
    assert plots._summary_fields(path) == {"alpha": 0.0}
    assert plots._summary_fields(tmp_path / "missing.json") == {}


def test_floor_runs_form_a_series_after_the_persona_mixture(tmp_path: Path) -> None:
    immediate = _steps(300, alpha=0.0, reward_kind="immediate")
    floor = _steps(300, alpha=0.0, reward_kind="immediate", reward_floor=FLOOR)
    cfg_floor = [{"step": 0, "config": {"reward": {"kind": "immediate", "floor": FLOOR}}}]
    assert plots._group_of_run("grpo-immediate", immediate) == 0.0
    assert plots._group_of_run("grpo-immediate-floor30", floor) == "reply_floor"
    assert plots._group_of_run("x", cfg_floor) == "reply_floor"
    assert plots._floor_of_run(immediate) is None
    grpo = {
        "grpo-immediate-floor30": floor,
        "grpo-immediate-floor30-s43": floor,
        "grpo-immediate": immediate,
        "grpo-persona-mix": _steps(300, **MIX),
        "grpo-delayed-a0.3": _steps(300, alpha=0.3),
    }
    groups = plots._group_runs(grpo)
    assert list(groups) == [0.0, 0.3, "persona_mix", "reply_floor"]
    assert list(groups["reply_floor"]) == ["grpo-immediate-floor30", "grpo-immediate-floor30-s43"]
    assert plots._group_label("reply_floor", 3) == "GRPO · α = 0 + reply floor (3 runs)"
    bon = [{"curves": {f"alpha={a}": {} for a in (0.0, 0.3, 0.6, 1.0)}}]
    colors = plots._series_colors(bon, grpo)
    assert list(colors) == [0.0, 0.3, 0.6, 1.0, "persona_mix", "reply_floor"]
    assert colors["persona_mix"] == plots.PALETTE[4]  # the floor takes the next slot
    assert colors["reply_floor"] == plots.PALETTE[5]
    _assert_png(plots.plot_goodhart_frontier([], grpo, tmp_path / "f.png"))
    _assert_png(plots.plot_grpo_curves(grpo, tmp_path / "c.png"))


def test_summary_fields_read_the_training_length(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"alpha": 0.0, "config": {"grpo": {"max_steps": 600}}}))
    assert plots._summary_fields(path) == {"alpha": 0.0, "max_steps": 600}
    path.write_text(json.dumps({"alpha": 0.0, "config": {"grpo": {}}}))
    assert plots._summary_fields(path) == {"alpha": 0.0}


def test_long_floor_runs_form_their_own_series_after_the_floor(tmp_path: Path) -> None:
    def summary(steps: int) -> dict:
        reward = {"kind": "immediate", "alpha": 0.0, "floor": FLOOR}
        return {
            "alpha": 0.0,
            "floor": FLOOR,
            "config": {"reward": reward, "grpo": {"max_steps": steps}},
        }

    _write_run(tmp_path, "grpo", "grpo-immediate", _steps(300), {"alpha": 0.0})
    _write_run(tmp_path, "grpo_600", "grpo-persona-mix", _steps(600), MIX)
    for seed in ("", "-s43"):
        _write_run(
            tmp_path, "grpo_floor", f"grpo-immediate-floor30{seed}", _steps(300), summary(300)
        )
        _write_run(
            tmp_path,
            "grpo_floor_600",
            f"grpo-immediate-floor30-600{seed}",
            _steps(600),
            summary(600),
        )
    grpo = plots.load_results(tmp_path)["grpo"]
    assert grpo["grpo-immediate-floor30-600"][0]["max_steps"] == 600
    groups = plots._group_runs(grpo)
    assert list(groups) == [0.0, "persona_mix", "reply_floor", "reply_floor_600"]
    assert list(groups["reply_floor_600"]) == [
        "grpo-immediate-floor30-600",
        "grpo-immediate-floor30-600-s43",
    ]
    assert len(groups["reply_floor"]) == 2
    assert plots._run_extent(groups["reply_floor_600"]) is None
    label = plots._group_label("reply_floor_600", 3)
    assert label == "GRPO · α = 0 + reply floor, 600 steps (3 runs)"
    cfg_only = [{"step": 0, "config": {"reward": {"floor": FLOOR}, "grpo": {"max_steps": 600}}}]
    assert plots._group_of_run("x", cfg_only) == "reply_floor_600"
    assert plots._group_of_run("x", _steps(300, reward_floor=FLOOR, max_steps=300)) == "reply_floor"
    # the new series takes the next palette slot, so the earlier series keep their colours
    bon = [{"curves": {f"alpha={a}": {} for a in (0.0, 0.3, 0.6, 1.0)}}]
    colors = plots._series_colors(bon, grpo)
    assert colors["persona_mix"] == plots.PALETTE[4]
    assert colors["reply_floor"] == plots.PALETTE[5]
    assert colors["reply_floor_600"] == plots.PALETTE[6]
    _assert_png(plots.plot_goodhart_frontier([], grpo, tmp_path / "f.png"))
    _assert_png(plots.plot_grpo_curves(grpo, tmp_path / "c.png"))


def test_legend_rows_are_balanced() -> None:
    assert [plots._legend_columns(n) for n in (1, 2, 5)] == [1, 2, 5]
    assert [plots._legend_columns(n) for n in (6, 7, 10, 11)] == [3, 4, 5, 4]


def test_longer_runs_extend_their_series(tmp_path: Path) -> None:
    runs = {
        "grpo-persona-mix": _steps(300, **MIX),
        "grpo-persona-mix-s43": _steps(300, **MIX),
        "grpo-persona-mix-s44": _steps(600, **MIX),
        "grpo-persona-mix-s45": _steps(600, **MIX),
        "grpo-persona-mix-s46": _steps(600, **MIX),
    }
    groups = plots._group_runs(runs)
    assert list(groups) == ["persona_mix"] and len(groups["persona_mix"]) == 5
    extent = plots._run_extent(groups["persona_mix"])
    assert extent == (3, 600)
    label = plots._group_label("persona_mix", 5, extent)
    assert label == "GRPO · persona mix (5 runs, 3 to step 600)"
    short = {k: runs[k] for k in ("grpo-persona-mix", "grpo-persona-mix-s43")}
    assert plots._run_extent(short) is None
    assert plots._run_extent({"a": _steps(600)}) is None
    agg = plots._aggregate(groups["persona_mix"], lambda r: plots._metric(r, "gold_accuracy"))
    assert [a[0] for a in agg] == [0, 100, 200, 300, 400, 500, 600]
    assert [a[4] for a in agg] == [5, 5, 5, 5, 3, 3, 3]
    _assert_png(plots.plot_grpo_curves(runs, tmp_path / "c.png"))


def test_place_labels_never_overlaps(tmp_path: Path) -> None:
    plt = plots._plt()
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1])
    fig.tight_layout()
    labels = [(0.5, 0.5, "N=1", True)] * 3  # one text at one point is drawn once
    labels += [(0.5, 0.5, f"step {s}", True) for s in (100, 200)]
    labels += [(0.5, 0.5, f"step {s}", False) for s in range(300, 1300, 100)]
    plots._place_labels(fig, ax, labels)
    texts = [t.get_text() for t in ax.texts]
    assert texts.count("N=1") == 1
    assert {"step 100", "step 200"} <= set(texts)
    assert len(texts) == len(plots._LABEL_OFFSETS)  # optional labels without room are left out
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in ax.texts]
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    plt.close(fig)


def test_frontier_labels_the_shared_start_once(tmp_path: Path, monkeypatch) -> None:
    texts: list[str] = []
    save = plots._save

    def capture(fig, out):
        texts.extend(t.get_text() for t in fig.axes[0].texts)
        return save(fig, out)

    monkeypatch.setattr(plots, "_save", capture)

    def run(gold_per_step: float, **extra) -> list[dict]:
        return [
            dict(r, gold_accuracy=0.65 + gold_per_step * r["step"], **extra) for r in _steps(300)
        ]

    grpo = {
        "grpo-immediate": run(-2e-4, alpha=0.0),
        "grpo-delayed-a0.3": run(-1e-4, alpha=0.3),
        "grpo-immediate-floor30": run(1e-4, alpha=0.0, reward_floor=FLOOR),
    }
    _assert_png(plots.plot_goodhart_frontier([], grpo, tmp_path / "f.png"))
    assert texts.count("step 0") == 1
    assert texts.count("step 300") == 3
