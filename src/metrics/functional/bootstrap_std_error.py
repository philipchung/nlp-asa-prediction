from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


@dataclass
class BootstrapMetricResult:
    # Value of metric with original data (without bootstrap)
    value: int | float | np.ndarray | torch.Tensor
    # Value of metric for each bootstrap iteration
    boot_values: list[int | float | np.ndarray | torch.Tensor]
    # Mean of all boot_values
    boot_mean: float | np.ndarray | torch.Tensor
    # Standard Deviation of all boot_values
    boot_std: float | np.ndarray | torch.Tensor
    # Model Predictions used to compute boot_values
    boot_preds: np.ndarray | torch.Tensor
    # Target Labels used to compute boot_values
    boot_target: np.ndarray | torch.Tensor


def wrap_bootstrap_std_error(
    metric_function: callable, boot_samples: int = 100000, boot_iterations: int = 1000
) -> callable:
    """Wrap metric function in `bootstrap_std_error_wrapper` and return the function.
    The resulting function has the same function signature as the original function as
    long as the original function accepts `preds` and `target` as input arguments"""
    return functools.partial(
        bootstrap_std_error_wrapper,
        metric_function=metric_function,
        boot_samples=boot_samples,
        boot_iterations=boot_iterations,
    )


def bootstrap_std_error_wrapper(
    metric_function: callable,
    boot_samples: int = 100000,
    boot_iterations: int = 1000,
    preds: Union[np.ndarray, torch.Tensor] = None,
    target: Union[np.ndarray, torch.Tensor] = None,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Computes Bootstrapped Standard Error for a Metric in addition to Metric Value.
    This is valid for any metric function as it uses a non-parametric bootstrap to
    estimate the standard error.  For default `boot_samples` and `boot_iterations`, this
    adds roughly 2 seconds to each metric computation.

    Args:
        metric_function (callable): Function used to compute metric. Should accept arguments
            `preds`, `target`, `num_classes`, and `**kwargs`.
        boot_samples (int, optional): Number of bootstrap samples. Defaults to 100000.
        boot_iterations (int, optional): Number of bootstrap iterations. Defaults to 1000.
        preds (Union[np.ndarray, torch.Tensor], optional): Tensor of predictions.
        target (Union[np.ndarray, torch.Tensor], optional): Tensor of labels.

    Returns:
        dict[str, torch.Tensor]:  Has the following keys:
            - value: value of metric computed using all of `preds` and `target
            - boot_mean: mean of the bootstrap metric values
                (this should be very close to "value", otherwise increase
                `boot_samples` and `boot_iterations`.)
            - boot_std: standard deviation of the bootstrap metric values
                (this is the standard error)
    """
    "Computes Bootstrapped Standard Error for a Metric in addition to Metric Value."
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    num_samples = len(preds)

    # Derive Bootstrap Datasets & Corresponding Metrics
    boot_preds_values = []
    boot_target_values = []
    boot_metric_values = []
    for _ in range(boot_iterations):
        # Select Bootstrap samples (sample from original preds & target w/ replacement)
        indices = torch.randint(low=0, high=num_samples, size=(boot_samples,))
        boot_preds = preds[indices]
        boot_target = target[indices]
        # Compute Metric on Bootstrap Dataset
        boot_metric_value = metric_function(
            preds=boot_preds, target=boot_target, **kwargs
        )
        # Cache Bootstrap Iteration
        boot_preds_values += [boot_preds]
        boot_target_values += [boot_target]
        boot_metric_values += [boot_metric_value]
    boot_metric_values = torch.stack(boot_metric_values)
    # Mean & St.Dev of Bootstrap Metrics
    boot_mean = boot_metric_values.mean(axis=0)
    boot_std = boot_metric_values.std(axis=0)

    # Compute On Original Dataset (no bootstrap)
    metric_value = metric_function(preds=preds, target=target, **kwargs)
    return BootstrapMetricResult(
        value=metric_value,
        boot_values=boot_metric_values,
        boot_mean=boot_mean,
        boot_std=boot_std,
        boot_preds=boot_preds_values,
        boot_target=boot_target_values,
    )


def unpack_bootstrap_metric_result(metric_dict: dict) -> dict:
    "If metric has bootstrapped standard errors computed, unpack BootstrapMetricResult."
    unpacked_metric_dict = {}
    for metric_name, metric_result in metric_dict.items():
        if isinstance(metric_result, BootstrapMetricResult):
            # Aggregate Metric (metric is 0-tensor containing an int or float)
            if metric_result.value.shape == torch.Size([]):
                unpacked_metric_dict[f"{metric_name}"] = metric_result.value
                # unpacked_metric_dict[f"{metric_name}_mean"] = metric_result.boot_mean
                unpacked_metric_dict[f"{metric_name}_SE"] = metric_result.boot_std
                unpacked_metric_dict[
                    f"{metric_name}_bootvalues"
                ] = metric_result.boot_values
                # unpacked_metric_dict[f"{metric_name}_bootpreds"] = metric_result.boot_preds
                # unpacked_metric_dict[f"{metric_name}_boottargets"] = metric_result.boot_targets

            # Classwise Metric (metric is 1-tensor of size num_classes)
            else:
                unpacked_metric_dict[f"{metric_name}"] = metric_result.value
                # unpacked_metric_dict[f"{metric_name}_mean"] = metric_result.boot_mean
                unpacked_metric_dict[f"{metric_name}_SE"] = metric_result.boot_std
                # Bootvalues: Split tensor (num_bootstraps, num_classes) -> [(num_bootstraps,), ... , (num_bootstraps,)]
                num_classes = len(metric_result.value)
                boot_values = metric_result.boot_values
                boot_values = torch.tensor_split(boot_values, num_classes, dim=1)
                boot_values = [x.squeeze() for x in boot_values]
                unpacked_metric_dict[f"{metric_name}_bootvalues"] = boot_values

        else:
            unpacked_metric_dict[metric_name] = metric_result
    return unpacked_metric_dict
