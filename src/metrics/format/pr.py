from __future__ import annotations

import copy
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def format_pr_curve(
    metrics: dict[str, Any],
    pr_metric_name: str = "PrecisionRecallCurve",
    auprc_metric_names: list[str] = None,
    class_labels: list = None,
) -> dict[str, Any]:
    for metric_name, metric_value in metrics.items():
        if metric_name == pr_metric_name:
            metrics[pr_metric_name] = format_plotly_pr_curve(
                metrics, pr_metric_name, auprc_metric_names, class_labels
            )
    return metrics


def format_plotly_pr_curve(
    metrics: dict[str, Any],
    pr_metric_name: str = "PrecisionRecallCurve",
    auprc_metric_names: list[str] = None,
    class_labels: list = None,
) -> dict[str, Any]:
    """
    Converts PR curve data into plotly format.
    Adds baseline precision for each class.
    Adds average precision (AUPRC) in legend for each class.
    Outputs plotly graph object.
    """
    pr_value = metrics[pr_metric_name]
    if auprc_metric_names is None:
        auprc_metrics = {
            k: v
            for k, v in metrics.items()
            if (
                ("AUPRC" in k.upper())
                and ("MACRO" not in k.upper())
                and ("MICRO" not in k.upper())
                and ("WEIGHTED" not in k.upper())
            )
        }
    else:
        auprc_metrics = {k: v for k, v in metrics.items() if k in auprc_metric_names}

    # Repackage Precision, Recall, AveragePrecision (AUPRC) for all classes
    precision, recall, thresholds = pr_value
    precisions = [x.cpu().numpy() for x in precision]
    recalls = [x.cpu().numpy() for x in recall]
    d = {
        "precision": dict(zip(class_labels, precisions)),
        "recall": dict(zip(class_labels, recalls)),
        "average_precision": {
            k.split("/")[1]: v.cpu().numpy() for k, v in auprc_metrics.items()
        },
    }
    # Compute Baseline Precisions (% positive examples in each class)
    supports = {k: v for k, v in metrics.items() if "Support" in k}
    num_examples_in_dataset = sum(supports.values())
    baseline_precisions = {
        k.split("/")[1]: (v / num_examples_in_dataset).cpu().numpy()
        for k, v in supports.items()
    }
    d = {**d, "baseline_precision": baseline_precisions}

    return pr_plotly_figure(**d)


def pr_plotly_figure(
    recall: dict[str, np.ndarray],
    precision: dict[str, np.ndarray],
    average_precision: dict[str, float] = None,
    baseline_precision: dict[str, np.ndarray] = None,
) -> go.Figure:
    """
    Create Precision-Recall Curve plot using plotly library.

    Args:
        recall: dict of {class: numpy array of recall at each threshold}
        precision: dict of {class: numpy array of precision at each threshold}
        average_precision: dict of {class: float of area under ROC curve for class}
        baseline_precision (optional): dict of {class: numpy array of precision for random classifier}

    Returns:
        Plotly figure that plots PR curves for each class as well as micro and macro
        averages on the same plot.  Average Precision is given in the legend.
        Display the figure object using fig.show().
    """
    # Get Default Color Sequence
    color_sequence = copy.deepcopy(px.colors.qualitative.Plotly)
    color_mapping = {}

    fig = go.Figure()

    # Plot Precision-Recall curves
    for label, average_precision_value in average_precision.items():
        # Assign color for each label
        color_mapping[label] = color_sequence.pop(0)

        # Average Precision in legend
        name = f"{label} (AUPRC={average_precision_value:.2f})"
        # PR Curve
        fig.add_trace(
            go.Scatter(
                x=recall[label],
                y=precision[label],
                name=name,
                mode="lines",
                line={"dash": "solid", "color": color_mapping[label]},
            )
        )
        # Optionally plot baselines for reference if provided
        if baseline_precision:
            bp = baseline_precision[label]
            baseline_name = f"{label} (Random Classifier Precision={bp:.2f})"
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[bp, bp],
                    name=baseline_name,
                    mode="lines",
                    line={"dash": "dot", "color": color_mapping[label]},
                )
            )
            # fig.add_hline(
            #     y=baseline_precision[label], name=baseline_name, line_dash="dash"
            # )

    # Generate Iso-F1 curves
    f_scores = np.linspace(0.2, 0.8, num=4)
    for f_score in f_scores:
        # Generate Iso-F1 Values
        x = np.geomspace(0.1, 1, num=300)
        eta = 1e-10
        y = f_score * x / (2 * x - f_score + eta)
        # Limit within bounds of plot
        x = x[(y >= 0) & (y <= 1)]
        y = y[(y >= 0) & (y <= 1)]

        name = f"F1={f_score:.2f}"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                showlegend=False,
                mode="lines",
                opacity=0.2,
                line={"color": "gray"},
            )
        )

        # Add Iso-F1 Text Annotations
        fig.add_annotation(x=x[0], y=1.025, text=name, showarrow=False)

    xaxis = {
        "title": {"text": "Recall"},
        "showgrid": True,
        "scaleanchor": "x",
        "scaleratio": 1,
    }
    yaxis = {
        "title": {"text": "Precision"},
        "showgrid": True,
        "constrain": "domain",
    }
    fig.update_layout(
        title={"text": "Precision-Recall Curve", "x": 0.5},
        xaxis=xaxis,
        yaxis=yaxis,
    )
    return fig
