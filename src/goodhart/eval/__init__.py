"""Answer parsing and metrics (gold accuracy, sycophancy, proxy reward)."""

from goodhart.eval.answer_parse import parse_answer
from goodhart.eval.metrics import bootstrap_ci, compute_metrics, records_to_frame

__all__ = ["bootstrap_ci", "compute_metrics", "parse_answer", "records_to_frame"]
