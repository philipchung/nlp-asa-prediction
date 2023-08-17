from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch


def unpack_classwise_metrics(
    metrics: dict[str, Any], class_labels: list = None
) -> dict[str, torch.Tensor]:
    """Flatten classwise metrics stored in tensor [class1, class2, ... class3]
    into individual metrics."""
    if class_labels is None:
        raise ValueError("Missing arg `class_labels`.")
    # Transform each item in tensor into its own metric with name "metric_name/class"
    reformatted_metrics = {}
    for metric_name, metric_values in metrics.items():
        if isinstance(
            metric_values, torch.Tensor
        ) and metric_values.shape == torch.Size([]):
            # metric_values is 0-tensor (int or float wrapped in tensor with no dimension)
            reformatted_metric = {metric_name: metric_values}
        else:
            # metric_values is 1-tensor or larger shape
            # Unpack Classwise Metric
            reformatted_metric = {}
            # Assumes metric_values are a 1-tensor or array with same length as class_labels
            for metric_value, label in zip(metric_values, class_labels):
                # If metric name has "_SE" (Standard Error) or "_bootvalues" suffix,
                # we split this out and keep it at the end of the final name
                m = metric_name.split("_")
                if len(m) == 2:
                    reformatted_metric |= {f"{m[0]}/{label}_{m[1]}": metric_value}
                else:
                    reformatted_metric |= {f"{metric_name}/{label}": metric_value}
        reformatted_metrics |= reformatted_metric
    return reformatted_metrics


def unpack_statscores(statscores: torch.Tensor) -> dict[str, torch.Tensor]:
    """Unpack (C,5) tensor into 5 (C,) tensors each corresponding to
    TP, FP, TN, FN, Support for all classes."""
    tp, fp, tn, fn, sup = list(statscores.T)
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "Support": sup}


def arraylike_to_tensor(x: Union[np.ndarray, torch.Tensor, list]) -> torch.Tensor:
    "Converts array or list-like data into torch tensor."
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, list):
        return torch.tensor(x)
    else:
        raise ValueError("Cannot convert argument to type torch.tensor.")
