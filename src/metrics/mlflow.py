from __future__ import annotations

from typing import Any
import logging

import matplotlib
import pandas as pd
import plotly
import torch

import mlflow

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def log_metrics(metrics: dict[str, Any], step: int = 0) -> None:
    "Logs float metrics, artifact figures, artifact tables with MLflow."
    unlogged_metrics = log_only_dataframes_and_figures(metrics, step)

    # Assume unlogged metrics are float or integer metric to be logged
    mlflow.log_metrics(unlogged_metrics, step=step)


def log_only_dataframes_and_figures(
    metrics: dict[str, Any], step: int = 0
) -> dict[str, Any]:
    """Logs only dataframes, artifact figures, artifact tables with MLflow.
    Any metrics not logged are returned."""
    unlogged_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, pd.DataFrame):
            # Convert DataFrame to JSON dictionary, Save as JSON Artifact
            try:
                metric_name = k.replace("/", "-")
                mlflow.log_dict(
                    dictionary=v.to_dict(),
                    artifact_file=f"metrics/{metric_name}/{metric_name}-step={step:04}.json",
                )
                log.debug(f"Logged DataFrame as JSON Artifact: {k}")
            except Exception as e:
                log.debug(f"Failed logging DataFrame: {k}.  Value: {v}")
                log.debug(f"Exception: {e}")
        elif isinstance(v, plotly.graph_objects.Figure):
            # Log Plotly Figure as HTML Artifact
            try:
                metric_name = k.replace("/", "-")
                mlflow.log_figure(
                    figure=v,
                    artifact_file=f"metrics/{metric_name}/{metric_name}-step={step:04}.html",
                )
                log.debug(f"Logged Plotly Figure as HTML Artifact: {k}")
            except Exception as e:
                log.debug(f"Failed logging Plotly Figure: {k}.  Value: {v}")
                log.debug(f"Exception: {e}")
        elif isinstance(v, matplotlib.figure.Figure):
            # Log Matplotlib Figure as PNG Artifact
            try:
                metric_name = k.replace("/", "-")
                mlflow.log_figure(
                    figure=v,
                    artifact_file=f"metrics/{metric_name}/{metric_name}-step={step:04}.png",
                )
                log.debug(f"Logged Matplotlib Figure as PNG Artifact: {k}")
            except Exception as e:
                log.debug(f"Failed logging Matplotlib Figure: {k}.  Value: {v}")
                log.debug(f"Exception: {e}")
        else:
            log.debug(f"Non-figure Metric: {k}.  Type: {type(v)}.")
            # Otherwise assume float or integer metric to be logged
            unlogged_metrics[k] = v

    # Once all matplotlib.pyplot objects have been logged, close all plots
    matplotlib.pyplot.close()

    return unlogged_metrics


def filter_nonnumeric_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    "Remove non-float or non-integer metrics."
    numeric_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, pd.DataFrame):
            continue
        if isinstance(v, plotly.graph_objects.Figure):
            continue
        if isinstance(v, matplotlib.figure.Figure):
            continue
        else:
            # Otherwise assume float or integer metric
            numeric_metrics[k] = v
    return numeric_metrics


def itemize_tensors(
    metrics: dict[str, Any], make_float: bool = False
) -> dict[str, Any]:
    "Convert torch tensors to standard python numbers."
    metrics = {
        k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()
    }
    if make_float:
        metrics = {k: float(v) for k, v in metrics.items()}
    return metrics
