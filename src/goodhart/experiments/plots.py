"""Figures from ``results/``: Best-of-N curves, the Goodhart frontier, GRPO curves, baseline.

Every function takes the loaded results (see :func:`load_results`) and writes one PNG.
Matplotlib is imported inside the functions on the Agg backend, so the module imports on a
box without it.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PALETTE = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
OPINIONS = ["none", "correct", "wrong"]

_ALPHA_KEY = re.compile(r"alpha\s*=\s*([0-9.eE+-]+)")
_ALPHA_RUN = re.compile(r"-a(\d+(?:\.\d+)?)$")
_STEP_KEYS = ("step", "global_step", "steps")


# ----------------------------------------------------------------------------- loading


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _summary_alpha(path: Path) -> float | None:
    """``alpha`` of a GRPO run from its ``summary.json`` (top level or ``config.reward``)."""
    if not path.is_file():
        return None
    meta = _read_json(path)
    alpha = meta.get("alpha")
    if alpha is None:
        reward = (meta.get("config") or {}).get("reward") or {}
        alpha = reward.get("alpha") if isinstance(reward, dict) else None
    return None if alpha is None else float(alpha)


def _row_step(row: dict[str, Any]) -> float:
    for key in _STEP_KEYS:
        if row.get(key) is not None:
            return float(row[key])
    return 0.0


def load_results(results_dir: str | Path) -> dict[str, Any]:
    """Read every results JSON under ``results_dir``.

    Returns ``{"baseline": [payload...], "bon": [payload...], "grpo": {run_name: [row...]}}``.
    Baseline and BoN payloads are sorted by model name; GRPO rows (``curve.jsonl`` per run
    directory) are sorted by step and carry the run's ``alpha`` from ``summary.json`` when
    the rows lack it. Missing directories give empty containers.
    """
    root = Path(results_dir)
    baseline = (
        [_read_json(p) for p in sorted((root / "baseline").glob("*.json"))]
        if (root / "baseline").is_dir()
        else []
    )
    bon = (
        [
            _read_json(p)
            for p in sorted((root / "bon").glob("*.json"))
            if not p.stem.endswith("_samples")
        ]
        if (root / "bon").is_dir()
        else []
    )
    grpo: dict[str, list[dict[str, Any]]] = {}
    if (root / "grpo").is_dir():
        for curve in sorted((root / "grpo").glob("*/curve.jsonl")):
            rows = _read_jsonl(curve)
            if not rows:
                continue
            alpha = _summary_alpha(curve.parent / "summary.json")
            if alpha is not None:
                for row in rows:
                    row.setdefault("alpha", alpha)
            grpo[curve.parent.name] = sorted(rows, key=_row_step)
    baseline.sort(key=lambda p: str(p.get("model", "")))
    bon.sort(key=lambda p: str(p.get("model", "")))
    return {"baseline": baseline, "bon": bon, "grpo": grpo}


# ----------------------------------------------------------------------------- helpers


def _metric(row: dict[str, Any] | None, key: str) -> float | None:
    """Look ``key`` up at the top level, then under ``metrics`` and ``val``."""
    if not isinstance(row, dict):
        return None
    for container in (row, row.get("metrics"), row.get("val")):
        if isinstance(container, dict) and container.get(key) is not None:
            return float(container[key])
    return None


def _training_reward(row: dict[str, Any]) -> float | None:
    """Mean reward the optimizer saw; ``proxy_mean`` when a run recorded nothing else."""
    value = _metric(row, "reward_mean")
    return value if value is not None else _metric(row, "proxy_mean")


def _heldout_sycophancy(row: dict[str, Any]) -> float | None:
    for key in ("sycophancy_eval", "heldout", "held_out"):
        value = _metric(row.get(key), "sycophancy_rate")
        if value is not None:
            return value
    for key in (
        "sycophancy_eval_sycophancy_rate",
        "heldout_sycophancy_rate",
        "sycophancy_rate@sycophancy_eval",
    ):
        if row.get(key) is not None:
            return float(row[key])
    return None


def _alpha_of_key(key: str) -> float | None:
    match = _ALPHA_KEY.search(key)
    if match:
        return float(match.group(1))
    try:
        return float(key)
    except ValueError:
        return None


def _alpha_of_run(run_name: str, rows: list[dict[str, Any]]) -> float | None:
    first = rows[0] if rows else {}
    if first.get("alpha") is not None:
        return float(first["alpha"])
    reward = first.get("config", {}).get("reward", {}) if isinstance(first, dict) else {}
    if isinstance(reward, dict) and reward.get("alpha") is not None:
        return float(reward["alpha"])
    match = _ALPHA_RUN.search(run_name)
    return float(match.group(1)) if match else None


def _sorted_alpha_keys(curves: dict[str, Any]) -> list[str]:
    return sorted(curves, key=lambda k: (_alpha_of_key(k) is None, _alpha_of_key(k) or 0.0, k))


def _sorted_n(curve: dict[str, Any]) -> list[str]:
    return sorted(curve, key=lambda k: int(k))


def _short_model(name: str) -> str:
    return str(name).rsplit("/", 1)[-1]


def _fmt_alpha(alpha: float | None) -> str:
    return "α=?" if alpha is None else f"α={alpha:g}"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": TEXT_SECONDARY,
            "axes.labelcolor": TEXT_PRIMARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _style_axes(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _line_kwargs(color: str, marker: str, dashed: bool = False) -> dict[str, Any]:
    return {
        "color": color,
        "marker": marker,
        "markersize": 6,
        "markeredgecolor": "white",
        "markeredgewidth": 1.0,
        "linewidth": 2,
        "linestyle": "--" if dashed else "-",
    }


def _annotate(ax, x: float, y: float, text: str) -> None:
    ax.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
        color=TEXT_SECONDARY,
    )


def _save(fig, out: str | Path) -> Path:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return out_path


def _figure_legend(fig, axes: Iterable[Any]) -> None:
    handles: dict[str, Any] = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(label, handle)
    if handles:
        fig.legend(
            list(handles.values()),
            list(handles),
            loc="lower center",
            ncol=min(5, len(handles)),
            bbox_to_anchor=(0.5, -0.02),
        )


# ----------------------------------------------------------------------------- plots


def plot_bon_curves(
    bon: list[dict[str, Any]], out: str | Path, alpha_key: str | None = None
) -> Path | None:
    """Proxy, gold and sycophancy against log2 N, one line per model.

    Selection uses ``alpha_key`` (default ``alpha=0.0`` when present, else the smallest α).
    Returns the written path, or ``None`` when there is nothing to plot.
    """
    if not bon:
        return None
    plt = _plt()
    panels = [
        ("proxy_mean", "proxy reward (1–10)"),
        ("gold_accuracy", "gold accuracy"),
        ("sycophancy_rate", "sycophancy rate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    chosen: set[str] = set()
    for idx, payload in enumerate(bon):
        curves = payload.get("curves", {})
        if not curves:
            continue
        if alpha_key in curves:
            key = alpha_key
        elif "alpha=0.0" in curves:
            key = "alpha=0.0"
        else:
            key = _sorted_alpha_keys(curves)[0]
        chosen.add(key)
        curve = curves[key]
        n_keys = _sorted_n(curve)
        ns = [int(n) for n in n_keys]
        color = PALETTE[idx % len(PALETTE)]
        marker = MARKERS[idx % len(MARKERS)]
        label = _short_model(payload.get("model", f"model {idx}"))
        for ax, (metric, _) in zip(axes, panels, strict=True):
            ys = [_metric(curve[n], metric) for n in n_keys]
            pts = [(n, y) for n, y in zip(ns, ys, strict=True) if y is not None]
            if pts:
                ax.plot(*zip(*pts, strict=True), label=label, **_line_kwargs(color, marker))
    all_n = sorted({int(n) for p in bon for c in p.get("curves", {}).values() for n in c})
    for ax, (_, ylabel) in zip(axes, panels, strict=True):
        _style_axes(ax)
        ax.set_xscale("log", base=2)
        if all_n:
            ax.set_xticks(all_n)
            ax.set_xticklabels([str(n) for n in all_n])
        ax.set_xlabel("N (log₂ scale)")
        ax.set_ylabel(ylabel)
    title = "Best-of-N" + (f" — selection by {', '.join(sorted(chosen))}" if chosen else "")
    fig.suptitle(title, color=TEXT_PRIMARY)
    _figure_legend(fig, axes)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _save(fig, out)


def plot_goodhart_frontier(
    bon: list[dict[str, Any]], grpo: dict[str, list[dict[str, Any]]], out: str | Path
) -> Path | None:
    """Gold accuracy (y) against proxy reward (x), one curve per α; points labelled N or step.

    Best-of-N curves are solid, GRPO runs dashed. Returns ``None`` when nothing is plottable.
    """
    if not bon and not grpo:
        return None
    plt = _plt()
    alphas: list[float | None] = []
    for payload in bon:
        alphas.extend(_alpha_of_key(k) for k in payload.get("curves", {}))
    for run_name, rows in grpo.items():
        alphas.append(_alpha_of_run(run_name, rows))
    alpha_order = sorted({a for a in alphas if a is not None}) + ([None] if None in alphas else [])
    color_of = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(alpha_order)}

    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = False
    for idx, payload in enumerate(bon):
        model = _short_model(payload.get("model", f"model {idx}"))
        marker = MARKERS[idx % len(MARKERS)]
        for key in _sorted_alpha_keys(payload.get("curves", {})):
            alpha = _alpha_of_key(key)
            curve = payload["curves"][key]
            pts = []
            for n in _sorted_n(curve):
                x, y = _metric(curve[n], "proxy_mean"), _metric(curve[n], "gold_accuracy")
                if x is not None and y is not None:
                    pts.append((int(n), x, y))
            if not pts:
                continue
            plotted = True
            ax.plot(
                [p[1] for p in pts],
                [p[2] for p in pts],
                label=f"{model} · {_fmt_alpha(alpha)} (BoN)",
                **_line_kwargs(color_of[alpha], marker),
            )
            for n, x, y in pts:
                _annotate(ax, x, y, f"N={n}")
    for idx, (run_name, rows) in enumerate(grpo.items()):
        alpha = _alpha_of_run(run_name, rows)
        pts = []
        for row in rows:
            x, y = _metric(row, "proxy_mean"), _metric(row, "gold_accuracy")
            if x is not None and y is not None:
                pts.append((_row_step(row), x, y))
        if not pts:
            continue
        plotted = True
        marker = MARKERS[(idx + len(bon)) % len(MARKERS)]
        ax.plot(
            [p[1] for p in pts],
            [p[2] for p in pts],
            label=f"{run_name} · {_fmt_alpha(alpha)} (GRPO)",
            **_line_kwargs(color_of[alpha], marker, dashed=True),
        )
        stride = max(1, len(pts) // 5)
        for i, (step, x, y) in enumerate(pts):
            if i == 0 or i == len(pts) - 1 or i % stride == 0:
                _annotate(ax, x, y, f"step {step:g}")
    if not plotted:
        plt.close(fig)
        return None
    _style_axes(ax)
    ax.set_xlabel("proxy reward (simulated user, 1–10)")
    ax.set_ylabel("gold accuracy")
    ax.set_title("Goodhart frontier: gold vs proxy as optimization strength grows")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.tight_layout()
    return _save(fig, out)


def plot_grpo_curves(grpo: dict[str, list[dict[str, Any]]], out: str | Path) -> Path | None:
    """Proxy, training reward, gold, sycophancy and the held-out metrics against step.

    One line per run. The proxy panel is the immediate judge; the training-reward panel is
    what the optimizer saw (the same thing for an immediate-reward run).
    """
    if not grpo:
        return None
    plt = _plt()
    panels = [
        ("proxy_mean", "proxy reward: immediate judge (1–10)"),
        ("reward", "training reward (1–10)"),
        ("gold_accuracy", "gold accuracy"),
        ("sycophancy_rate", "sycophancy rate"),
        ("heldout", "held-out sycophancy rate (SycophancyEval)"),
        ("heldout_gold_accuracy", "held-out gold accuracy (SycophancyEval)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.ravel()
    for idx, (run_name, rows) in enumerate(grpo.items()):
        steps = [_row_step(r) for r in rows]
        color = PALETTE[idx % len(PALETTE)]
        marker = MARKERS[idx % len(MARKERS)]
        for ax, (metric, _) in zip(axes, panels, strict=True):
            if metric == "heldout":
                ys = [_heldout_sycophancy(r) for r in rows]
            elif metric == "reward":
                ys = [_training_reward(r) for r in rows]
            else:
                ys = [_metric(r, metric) for r in rows]
            pts = [(s, y) for s, y in zip(steps, ys, strict=True) if y is not None]
            if pts:
                ax.plot(*zip(*pts, strict=True), label=run_name, **_line_kwargs(color, marker))
    for ax, (_, ylabel) in zip(axes, panels, strict=True):
        _style_axes(ax)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        if not ax.lines:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color=TEXT_SECONDARY)
    fig.suptitle("GRPO against the proxy reward", color=TEXT_PRIMARY)
    _figure_legend(fig, axes)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, out)


def plot_baseline(baseline: list[dict[str, Any]], out: str | Path) -> Path | None:
    """Gold accuracy per opinion type, grouped by model."""
    if not baseline:
        return None
    plt = _plt()
    models = [_short_model(p.get("model", f"model {i}")) for i, p in enumerate(baseline)]
    per_opinion = {
        op: [_metric(p.get("metrics"), f"gold_accuracy@{op}") for p in baseline] for op in OPINIONS
    }
    series = [(op, vals) for op, vals in per_opinion.items() if any(v is not None for v in vals)]
    if not series:
        series = [("all", [_metric(p.get("metrics"), "gold_accuracy") for p in baseline])]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(models) + 2), 4.2))
    width = 0.8 / len(series)
    for j, (op, vals) in enumerate(series):
        xs = [i + (j - (len(series) - 1) / 2) * width for i in range(len(models))]
        ys = [v if v is not None else 0.0 for v in vals]
        ax.bar(xs, ys, width=width * 0.9, color=PALETTE[j % len(PALETTE)], label=f"opinion={op}")
        for x, v in zip(xs, vals, strict=True):
            if v is not None:
                ax.text(
                    x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY
                )
    _style_axes(ax)
    ax.grid(False, axis="x")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("gold accuracy")
    ax.set_title("Baseline: gold accuracy per opinion type")
    ax.legend(loc="upper center", ncol=len(series), fontsize=8, frameon=False)
    fig.tight_layout()
    return _save(fig, out)


# ----------------------------------------------------------------------------- CLI


def make_all(results_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Build every figure that has data and return the written paths."""
    results = load_results(results_dir)
    out = Path(out_dir)
    written = [
        plot_bon_curves(results["bon"], out / "bon_curves.png"),
        plot_goodhart_frontier(results["bon"], results["grpo"], out / "goodhart_frontier.png"),
        plot_grpo_curves(results["grpo"], out / "grpo_curves.png"),
        plot_baseline(results["baseline"], out / "baseline.png"),
    ]
    return [p for p in written if p is not None]


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description="Render figures from results/ into figures/.")
    parser.add_argument("--results", default="results", help="directory with results JSONs")
    parser.add_argument("--out", default="figures", help="directory for PNG files")
    args = parser.parse_args(argv)
    written = make_all(args.results, args.out)
    for path in written:
        print(f"wrote {path}")
    if not written:
        print(f"no results found under {args.results}")
    return written


if __name__ == "__main__":
    main()
