from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go


def format_roc(
    metrics: dict[str, Any],
    roc_metric_name: str = "ROC",
    auroc_metric_names: list[str] = None,
    class_labels: list = None,
) -> dict[str, Any]:
    for metric_name, metric_value in metrics.items():
        if metric_name == roc_metric_name:
            metrics[roc_metric_name] = format_plotly_roc_curve(
                metrics, roc_metric_name, auroc_metric_names, class_labels
            )
    return metrics


def format_plotly_roc_curve(
    metrics: dict[str, Any],
    roc_metric_name: str = "ROC",
    auroc_metric_names: list[str] = None,
    class_labels: list = None,
) -> go.Figure:
    """
    Converts ROC curve data into plotly format.
    Adds AUROC in legend for each class.
    Outputs plotly graph object.
    """
    roc_value = metrics[roc_metric_name]
    if auroc_metric_names is None:
        auroc_metrics = {
            k: v
            for k, v in metrics.items()
            if (
                ("AUROC" in k.upper())
                and ("MACRO" not in k.upper())
                and ("MICRO" not in k.upper())
                and ("WEIGHTED" not in k.upper())
                and ("AUCMU" not in k.upper())
            )
        }
    else:
        auroc_metrics = {k: v for k, v in metrics.items() if k in auroc_metric_names}

    # Repackage FPR, TPR, AUROC for all classes
    fpr, tpr, thresholds = roc_value
    fprs = [x.cpu().numpy() for x in fpr]
    tprs = [x.cpu().numpy() for x in tpr]
    d = {
        "fpr": dict(zip(class_labels, fprs)),
        "tpr": dict(zip(class_labels, tprs)),
        "auroc": {k.split("/")[1]: v.cpu().numpy() for k, v in auroc_metrics.items()},
    }
    return roc_plotly_figure(**d)


def roc_plotly_figure(
    fpr: dict[str, np.ndarray],
    tpr: dict[str, np.ndarray],
    auroc: dict[str, float],
) -> go.Figure:
    """
    Create ROC plot using plotly library.

    Args:
        fpr: dict of {class: numpy array of false positive rates}
        tpr: dict of {class: numpy array of true positive rates}
        auroc: dict of {class: float of area under ROC curve for class}

    Returns:
        Plotly figure that plots ROC curves for each class as well as micro and macro
        averages on the same plot.  AUC is given in the legend.  Display the figure object
        using fig.show().
    """
    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    for label, auroc_value in auroc.items():
        name = f"{label} (AUROC={auroc_value:.2f})"
        fig.add_trace(go.Scatter(x=fpr[label], y=tpr[label], name=name, mode="lines"))

    xaxis = {
        "title": {"text": "False Positive Rate"},
        "showgrid": True,
        "scaleanchor": "x",
        "scaleratio": 1,
    }
    yaxis = {
        "title": {"text": "True Positive Rate"},
        "showgrid": True,
        "constrain": "domain",
    }
    fig.update_layout(
        title={"text": "Receiver Operator Characteristic Curve", "x": 0.5},
        xaxis=xaxis,
        yaxis=yaxis,
    )
    return fig
