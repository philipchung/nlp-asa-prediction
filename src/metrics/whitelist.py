from __future__ import annotations

from typing import Any, Union

from omegaconf import DictConfig, OmegaConf

from src.utils.list_utils import flatten_list_of_list


def filter_by_whitelist(
    metrics: dict[str, Any],
    class_labels: list,
    whitelist: Union[list, dict, DictConfig],
) -> dict[str, Any]:
    """Filter metrics based on whitelist.

    Assumes all whitelist names are unique with exception of classwise metric
    names which may also appear in aggregate metric names.

    Args:
        metrics (dict): dictionary of metrics {metric name: metric value}
        class_labels (list): list of class label names for classwise metrics.
            Internal to this function, we expand classwise metrics specified in
            whitelist by these classes, then match by the expanded name.
        whitelist (list, dict, DictConfig): if list is passed in, it is used as
            the whitelist.  If hierarchical dict or DictConfig passed in, then
            will flatten hierarchy and use values.

    """
    # If whitelist is not a list, transform into a list
    if not isinstance(whitelist, list):
        whitelist = expand_classwise_whitelist(metrics, class_labels, whitelist)

    # Apply whitelist filter to metrics
    metrics = {k: v for k, v in metrics.items() if k in whitelist}
    return metrics


def expand_classwise_whitelist(
    class_labels: list,
    whitelist: Union[dict, DictConfig],
) -> dict | DictConfig:
    """Expands classwise names in whitelist by appending `/class_label` to it.
    Non-classwise names are not manipulated.
    This method only works for dict whitelists, supporting depth of 2.
    Only values at depth 2 are treated as whitelist.
    """
    # If DictConfig, transform into dict
    if isinstance(whitelist, DictConfig):
        whitelist = OmegaConf.to_container(whitelist, resolve=True)

    if isinstance(whitelist, dict):
        # Expand classwise metric names
        if "classwise" in whitelist:
            classwise_whitelist = whitelist.pop("classwise")
            new_classwise_whitelist = []
            if classwise_whitelist is not None:
                for classwise_metric_name in classwise_whitelist:
                    for label in class_labels:
                        new_classwise_name = classwise_metric_name + f"/{label}"
                        new_classwise_whitelist += [new_classwise_name]
        # Keep other metric names the same
        other_whitelist = flatten_list_of_list(v for k, v in whitelist.items())
        whitelist = other_whitelist + new_classwise_whitelist
    else:
        raise ValueError("Unknown type for argument `whitelist`.")
    return whitelist


def flatten_whitelist_dict(whitelist: Union[dict, DictConfig]) -> list:
    "Flatten values in dict into a list."
    # If DictConfig, transform into dict
    if isinstance(whitelist, DictConfig):
        whitelist = OmegaConf.to_container(whitelist, resolve=True)
    return flatten_list_of_list(v for k, v in whitelist.items())
