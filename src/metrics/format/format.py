from __future__ import annotations

from typing import Any, Union

from src.metrics.format.confusion_matrix import format_confusion_matrix
from src.metrics.format.pr import format_pr_curve
from src.metrics.format.roc import format_roc


def format_all(
    metrics: dict[str, Any],
    class_labels: list[str],
    split: str,
    target_name: str,
):
    metrics = format_roc(metrics, class_labels=class_labels)
    metrics = format_pr_curve(metrics, class_labels=class_labels)
    metrics = format_confusion_matrix(metrics, class_labels=class_labels)
    metrics = format_examples_table(metrics)
    metrics = format_metric_names(metrics, split=split, target_name=target_name)
    return metrics


def format_metric_names(metrics: dict[str, Any], split: str, target_name: str):
    """
    Formats metrics dict and renames by `split` & `target_name`.

    Args:
        metrics: dictionary of metrics with string key
        split: name of data split {"train", "validation", "test", "predict"}
        target_name: name of output variable (e.g. "ASA", "Emergency")

    Returns:
        metrics dict with formatting applied to keys (metric names).
    """
    split = split.capitalize()
    return {f"{target_name}/{split}/{k}": v for k, v in metrics.items()}


def format_examples_table(
    metrics: dict[str, Any], columns: Union[str, list] = None
) -> dict[str, Any]:
    "Select subset of columns, reorder & rename."
    if columns is None:
        # Default columns
        columns = [
            "HPI",
            "input_text",
            "ASA",
            "asa_class",
            "emergency",
            "predicted_label_name",
            "Probabilities",
            "Logits",
            "Loss",
            "ProcedureID",
            "PersonID",
        ]
    elif isinstance(columns, str):
        columns = [columns]
    elif isinstance(columns, list):
        columns = columns
    else:
        raise ValueError("Unknown type for argument `columns`.")
    # Apply col subset
    for k, v in metrics.items():
        if "Examples" in k:
            df = v
            col_names = list(set(columns).intersection(df.columns))
            metrics[k] = df.loc[:, col_names]
    return metrics
