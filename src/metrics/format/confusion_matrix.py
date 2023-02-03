from __future__ import annotations

from typing import Any, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def format_confusion_matrix(
    metrics: dict[str, Any],
    class_labels: list,
    confusion_matrix_metric_name: str = "ConfusionMatrix",
) -> dict[str, Any]:
    formatted_metrics = {}
    for metric_name, metric_value in metrics.items():
        if confusion_matrix_metric_name in metric_name:
            if "Normalize" in metric_name:
                # Plot Confusion Matrix Normalized to Percents
                if "NormalizeOverTargets" in metric_name:
                    title = "Confusion Matrix: Normalize Over Actual"
                elif "NormalizeOverPredictions" in metric_name:
                    title = "Confusion Matrix: Normalize Over Predicted"
                elif "NormalizeOverAll" in metric_name:
                    title = "Confusion Matrix: Normalize Over All"
                else:
                    title = "Confusion Matrix"

                formatted_metrics[metric_name] = confusion_matrix_plot(
                    confusion_matrix=metric_value,
                    class_labels=class_labels,
                    title=title,
                    font_scale=1.1,
                    fig_size=(7, 5),
                    format_percent=True,
                )
            else:
                # Plot Raw Confusion Matrix with Counts
                formatted_metrics[metric_name] = confusion_matrix_plot(
                    confusion_matrix=metric_value,
                    class_labels=class_labels,
                    font_scale=1.1,
                    fig_size=(7, 5),
                    format_percent=False,
                )
        else:
            formatted_metrics[metric_name] = metric_value
    return formatted_metrics


def confusion_matrix_plot(
    confusion_matrix: Union[torch.tensor, np.ndarray],
    class_labels: Union[list, tuple],
    title: str = "Confusion Matrix",
    font_scale: int = 1.2,
    fig_size: tuple = (7, 5),
    cmap: str = "Blues",
    format_percent: bool = False,
) -> matplotlib.figure.Figure:
    "Creates confusion matrix plot."
    # Ensure tensors are on CPU
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.cpu()
    df_cm = pd.DataFrame(confusion_matrix, columns=class_labels, index=class_labels)
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"

    fig = plt.figure(figsize=fig_size)
    sns.set(font_scale=font_scale)
    if format_percent:
        sns.heatmap(df_cm, cmap=cmap, annot=True, fmt=".2%").set(title=title)
    else:
        sns.heatmap(df_cm, cmap=cmap, annot=True, fmt="g").set(title="Confusion Matrix")
    return fig
