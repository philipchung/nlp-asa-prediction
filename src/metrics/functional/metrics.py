from __future__ import annotations

import logging
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.nn.functional import softmax
from torchmetrics.functional.classification import (
    multiclass_auroc,
    multiclass_average_precision,
    multiclass_cohen_kappa,
    multiclass_confusion_matrix,
    multiclass_f1_score,
    multiclass_hinge_loss,
    multiclass_matthews_corrcoef,
    multiclass_precision,
    multiclass_precision_recall_curve,
    multiclass_recall,
    multiclass_roc,
    multiclass_stat_scores,
)

from src.metrics.functional.bootstrap_std_error import (
    unpack_bootstrap_metric_result,
    wrap_bootstrap_std_error,
)
from src.metrics.functional.loss import compute_cross_entropy_loss
from src.metrics.functional.top_examples import easiest_hardest_examples
from src.metrics.metrics.auc_mu import multiclass_aucmu
from src.metrics.utils import (
    arraylike_to_tensor,
    unpack_classwise_metrics,
    unpack_statscores,
)
from src.metrics.whitelist import flatten_whitelist_dict

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


def classwise_statistic_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    class_labels: list = None,
    bootstrap_std_error: bool = True,
    metric_names: list[str] = ["StatScores", "Precision", "Recall", "F1"],
) -> dict[str, torch.Tensor]:
    preds = arraylike_to_tensor(preds)
    target = arraylike_to_tensor(target)

    # Optionally Compute Version of Metric w/ Bootstrapped Standard Errors
    if bootstrap_std_error:
        multiclass_precision_fn = wrap_bootstrap_std_error(multiclass_precision)
        multiclass_recall_fn = wrap_bootstrap_std_error(multiclass_recall)
        multiclass_f1_score_fn = wrap_bootstrap_std_error(multiclass_f1_score)
    else:
        multiclass_precision_fn = multiclass_precision
        multiclass_recall_fn = multiclass_recall
        multiclass_f1_score_fn = multiclass_f1_score

    # StatScores computes TP/FP/TN/FN/Support all at once
    if any(x in ("TP", "FP", "TN", "FN", "Support") for x in metric_names):
        metric_names = [
            x for x in metric_names if x not in ("TP", "FP", "TN", "FN", "Support")
        ]
        metric_names += ["StatScores"]

    # Compute each selected metric
    metrics = {}
    for metric_name in metric_names:
        match metric_name:
            case "StatScores":
                # Statscores (TP,FP,TN,FN,Support) outputs tensor shape (C,5),
                # with C=num_classes.
                statscores = multiclass_stat_scores(
                    preds=preds, target=target, num_classes=num_classes, average=None
                )
                # Convert to {"TP: [class0, class1, ..., classN]",
                # "FP": [...],
                # "TN": [...],
                # "FN": [...],
                # "Support": [...]}
                statscores_dict = unpack_statscores(statscores)
                metrics |= statscores_dict
            case "Precision":
                metrics |= {
                    "Precision": multiclass_precision_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average=None,
                    )
                }
            case "Recall":
                metrics |= {
                    "Recall": multiclass_recall_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average=None,
                    )
                }
            case "F1":
                metrics |= {
                    "F1": multiclass_f1_score_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average=None,
                    )
                }

    # Flatten Nested Bootstrap Results
    metrics = unpack_bootstrap_metric_result(metrics)
    # For metrics with tensor shape (C,) output, unpack each item in tensor
    # into its own metric and label with class labels.
    metrics = unpack_classwise_metrics(metrics, class_labels)
    return metrics


def class_aggregate_statistic_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    bootstrap_std_error: bool = True,
    metric_names: list[str] = [
        "Precision/Micro",
        "Precision/Macro",
        "Recall/Micro",
        "Recall/Macro",
        "F1/Micro",
        "F1/Macro",
        "MCC/MCC",
    ],
) -> dict[str, torch.Tensor]:
    preds = arraylike_to_tensor(preds)
    target = arraylike_to_tensor(target)

    # Optionally Compute Version of Metric w/ Bootstrapped Standard Errors
    if bootstrap_std_error:
        multiclass_precision_fn = wrap_bootstrap_std_error(multiclass_precision)
        multiclass_recall_fn = wrap_bootstrap_std_error(multiclass_recall)
        multiclass_f1_score_fn = wrap_bootstrap_std_error(multiclass_f1_score)
        multiclass_matthews_corrcoef_fn = wrap_bootstrap_std_error(
            multiclass_matthews_corrcoef
        )
        multiclass_cohen_kappa_fn = wrap_bootstrap_std_error(multiclass_cohen_kappa)
    else:
        multiclass_precision_fn = multiclass_precision
        multiclass_recall_fn = multiclass_recall
        multiclass_f1_score_fn = multiclass_f1_score
        multiclass_matthews_corrcoef_fn = multiclass_matthews_corrcoef
        multiclass_cohen_kappa_fn = multiclass_cohen_kappa

    # Compute each selected metric
    metrics = {}
    for metric_name in metric_names:
        match metric_name:
            case "Precision/Micro":
                metrics |= {
                    "Precision/Micro": multiclass_precision_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average="micro",
                    )
                }
            case "Precision/Macro":
                metrics |= {
                    "Precision/Macro": multiclass_precision_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average="macro",
                    )
                }
            case "Precision/Weighted":
                metrics |= {
                    "Precision/Weighted": multiclass_precision_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average="weighted",
                    )
                }
            case "Recall/Micro":
                metrics |= {
                    "Recall/Micro": multiclass_recall_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average="micro",
                    )
                }
            case "Recall/Macro":
                metrics |= {
                    "Recall/Macro": multiclass_recall_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average="macro",
                    )
                }
            case "Recall/Weighted":
                metrics |= {
                    "Recall/Weighted": multiclass_recall_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average="weighted",
                    )
                }
            case "F1/Micro":
                metrics |= {
                    "F1/Micro": multiclass_f1_score_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average="micro",
                    )
                }
            case "F1/Macro":
                metrics |= {
                    "F1/Macro": multiclass_f1_score_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average="macro",
                    )
                }
            case "F1/Weighted":
                metrics |= {
                    "F1/Weighted": multiclass_f1_score_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        average="weighted",
                    )
                }
            case "MCC/MCC":
                metrics |= {
                    "MCC/MCC": multiclass_matthews_corrcoef_fn(
                        preds=preds, target=target, num_classes=num_classes
                    )
                }
            case "CohenKappa/Unweighted":
                metrics |= {
                    "CohenKappa/Unweighted": multiclass_cohen_kappa_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        weights=None,
                    )
                }
            case "CohenKappa/WeightedLinear":
                metrics |= {
                    "CohenKappa/WeightedLinear": multiclass_cohen_kappa_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        weights="linear",
                    )
                }
            case "CohenKappa/WeightedQuadratic":
                metrics |= {
                    "CohenKappa/WeightedQuadratic": multiclass_cohen_kappa_fn(
                        preds=preds,
                        target=target,
                        num_classes=num_classes,
                        weights="quadratic",
                    )
                }
    # Flatten Nested Bootstrap Results
    metrics = unpack_bootstrap_metric_result(metrics)
    return metrics


def roc_metrics(
    pred_proba: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    class_labels: list = None,
    bootstrap_std_error: bool = True,
    metric_names: list[str] = [
        "ROC",
        "AUROC",
        "AUROC/Macro",
        "AUROC/AUCmu",
    ],
) -> dict[str, torch.Tensor]:
    pred_proba = arraylike_to_tensor(pred_proba)
    target = arraylike_to_tensor(target)

    # Optionally Compute Version of Metric w/ Bootstrapped Standard Errors
    if bootstrap_std_error:
        multiclass_auroc_fn = wrap_bootstrap_std_error(multiclass_auroc)
        multiclass_aucmu_fn = wrap_bootstrap_std_error(multiclass_aucmu)

    else:
        multiclass_auroc_fn = multiclass_auroc
        multiclass_aucmu_fn = multiclass_aucmu

    # Compute each selected metric
    roc_curve = {}
    if "ROC" in metric_names:
        # Compute ROC Curve separately since we don't want to unpack the results
        roc_curve |= {
            "ROC": multiclass_roc(
                preds=pred_proba,
                target=target,
                num_classes=num_classes,
                thresholds=None,
            )
        }
        metric_names.remove("ROC")

    metrics = {}
    for metric_name in metric_names:
        match metric_name:
            case "AUROC":
                m = {
                    "AUROC": multiclass_auroc_fn(
                        preds=pred_proba,
                        target=target,
                        num_classes=num_classes,
                        average=None,
                        thresholds=None,
                    )
                }
                # Flatten Nested Bootstrap Results
                m = unpack_bootstrap_metric_result(m)
                # For metrics with tensor shape (C,) output, unpack each item in tensor
                # into its own metric and label with class labels.
                m = unpack_classwise_metrics(m, class_labels)
                metrics |= m
            case "AUROC/Macro":
                m = {
                    "AUROC/Macro": multiclass_auroc_fn(
                        preds=pred_proba,
                        target=target,
                        num_classes=num_classes,
                        average="macro",
                        thresholds=None,
                    )
                }
                # Flatten Nested Bootstrap Results
                m = unpack_bootstrap_metric_result(m)
                metrics |= m
            case "AUROC/Weighted":
                m = {
                    "AUROC/Weighted": multiclass_auroc_fn(
                        preds=pred_proba,
                        target=target,
                        num_classes=num_classes,
                        average="weighted",
                        thresholds=None,
                    )
                }
                # Flatten Nested Bootstrap Results
                m = unpack_bootstrap_metric_result(m)
                metrics |= m
            case "AUROC/AUCmu":
                m = {
                    "AUROC/AUCmu": multiclass_aucmu_fn(preds=pred_proba, target=target)
                }
                # Flatten Nested Bootstrap Results
                m = unpack_bootstrap_metric_result(m)
                metrics |= m

    metrics = roc_curve | metrics
    return metrics


def pr_metrics(
    pred_proba: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    class_labels: list = None,
    bootstrap_std_error: bool = True,
    metric_names: list[str] = [
        "PrecisionRecallCurve",
        "AUPRC",
        "AUPRC/Macro",
    ],
) -> dict[str, torch.Tensor]:
    pred_proba = arraylike_to_tensor(pred_proba)
    target = arraylike_to_tensor(target)

    # Optionally Compute Version of Metric w/ Bootstrapped Standard Errors
    if bootstrap_std_error:
        multiclass_average_precision_fn = wrap_bootstrap_std_error(
            multiclass_average_precision
        )

    else:
        multiclass_average_precision_fn = multiclass_average_precision

    # Compute each selected metric
    pr_curve = {}
    if "PrecisionRecallCurve" in metric_names:
        # Compute PrecisionRecallCurve separately since we don't want to unpack the results
        pr_curve |= {
            "PrecisionRecallCurve": multiclass_precision_recall_curve(
                preds=pred_proba,
                target=target,
                num_classes=num_classes,
                thresholds=None,
            )
        }
        metric_names.remove("PrecisionRecallCurve")

    metrics = {}
    for metric_name in metric_names:
        match metric_name:
            case "AUPRC":
                m = {
                    "AUPRC": multiclass_average_precision_fn(
                        preds=pred_proba,
                        target=target,
                        num_classes=num_classes,
                        average=None,
                        thresholds=None,
                    ),
                }
                # Flatten Nested Bootstrap Results
                m = unpack_bootstrap_metric_result(m)
                # For metrics with tensor shape (C,) output, unpack each item in tensor
                # into its own metric and label with class labels.
                m = unpack_classwise_metrics(m, class_labels)
                metrics |= m
            case "AUPRC/Macro":
                m = {
                    "AUPRC/Macro": multiclass_average_precision_fn(
                        preds=pred_proba,
                        target=target,
                        num_classes=num_classes,
                        average="macro",
                        thresholds=None,
                    )
                }
                # Flatten Nested Bootstrap Results
                m = unpack_bootstrap_metric_result(m)
                metrics |= m
            case "AUPRC/Weighted":
                m = {
                    "AUPRC/Weighted": multiclass_average_precision_fn(
                        preds=pred_proba,
                        target=target,
                        num_classes=num_classes,
                        average="weighted",
                        thresholds=None,
                    )
                }
                # Flatten Nested Bootstrap Results
                m = unpack_bootstrap_metric_result(m)
                metrics |= m

    metrics = pr_curve | metrics
    return metrics


def confusion_matrix_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
) -> dict[str, torch.Tensor]:
    """
    Confusion matrix metrics.

    Args:
        preds (numpy array or torch tensor): predicted class for each example
        target (numpy array or torch tensor): target class for each example
        num_classes (int): number of classes

    Returns:
        Dict of confusion matricies with different normalizations.
        Each confusion matrix has shape (C,C) with rows corresponding
        to predicted class and columns corresponding to target class.
        (e.g. value at (1, 3) are examples predicted class 1 but
        actual class is 3)
    """
    preds = arraylike_to_tensor(preds)
    target = arraylike_to_tensor(target)
    confusion_matrix_metrics = {
        "ConfusionMatrix/Raw": multiclass_confusion_matrix(
            preds=preds, target=target, num_classes=num_classes, normalize=None
        ),
        "ConfusionMatrix/NormalizeOverTargets": multiclass_confusion_matrix(
            preds=preds, target=target, num_classes=num_classes, normalize="true"
        ),
        "ConfusionMatrix/NormalizeOverPredictions": multiclass_confusion_matrix(
            preds=preds, target=target, num_classes=num_classes, normalize="pred"
        ),
        "ConfusionMatrix/NormalizeOverAll": multiclass_confusion_matrix(
            preds=preds, target=target, num_classes=num_classes, normalize="all"
        ),
    }
    return confusion_matrix_metrics


def all_metrics(
    preds: Union[np.ndarray, torch.Tensor] = None,
    pred_proba: Union[np.ndarray, torch.Tensor] = None,
    pred_score: Union[np.ndarray, torch.Tensor] = None,
    target: Union[np.ndarray, torch.Tensor] = None,
    score_kind: str = "probabilities",
    include_loss: bool = True,
    num_classes: int = 4,
    class_labels: list = None,
    top_k_easiest_hardest: int | None = None,
    dataset: pd.DataFrame = None,
    id2label: dict[int, str] = None,
    whitelist: Union[list[str], dict[str, Any], DictConfig[str, Any]] = None,
    bootstrap_std_error: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute loss and metrics.

    The loss type to compute depends on the objective function of the model that
    you are evaluating and whether it returns raw scores or probabilities.  In general,
    "hinge" loss is used for support vector machines and its calculation requires
    providing `pred_score` argument.  "cross_entropy" loss is used for most other
    models and is a multiclass generalization of loglikelihood, and this function
    expects `pred_proba` argument.  It is possible to compute "cross_entropy" loss for
    support vector machine models as an evaluation metric if output scores are
    normalized to probabilities first.

    Args:
        preds (numpy array or torch tensor): (N,) predicted classes for each example
        pred_proba (numpy array or torch tensor): (N,C) with values corresponding
            to the probabilities for predicting each class C.  Note that a pred_score
            can be transformed to pred_proba via passing values through softmax.
        pred_score (numpy array or torch tensor): (N,C) with values corresponding
            to the score for predicting each class C.  This is the raw score outputs
            from the objective function and has not been normalized into probabilities.
        target (numpy array or torch tensor): target class for each example
        score_kind (str): either "probabilities" or "logits" which specifies
            the kind of output a model produces.  This affects which loss function
            is needed to compute cross entropy loss over the input scores.
        include_loss (bool): whether or not to include loss in output metrics dictionary
        num_classes (int): number of classes
        class_labels (list): list of string class labels
        top_k_easiest_hardest (int): top k easiest and top k hardest examples in
            dataset to visualize in table.  If `None`, then skip this metric.
        dataset (pd.DataFrame): dataset from which top k easiest and hardest examples
            are taken from.  This dataset should have length (N,) and order of examples
            should be conserved with target, pred_proba and pred_score.
        id2label (dict): dictionary that maps class ID to string label
        whitelist (list, dict, or DictConfig): Metrics to be included.
        bootstrap_std_error (bool):  Whether to compute standard error of metrics.
            This takes roughly an additional 2-5 seconds per metric.

    Returns:
        Dict of metrics.
    """
    preds = arraylike_to_tensor(preds)
    target = arraylike_to_tensor(target)
    if pred_proba is None and pred_score is None:
        raise ValueError("Must provide either `pred_proba` or `pred_score`.")
    if pred_score is not None:
        pred_score = arraylike_to_tensor(pred_score)
        # If pred_proba not given along with pred_score, we can compute one using
        # softmax normalization.  pred_proba is still needed for computing
        # roc and pr curves.
        if pred_proba is None:
            pred_proba = softmax(pred_score, dim=1)
        else:
            pred_proba = arraylike_to_tensor(pred_proba)

    if isinstance(whitelist, dict) or isinstance(whitelist, DictConfig):
        whitelist = flatten_whitelist_dict(whitelist)

    metrics = {}
    if include_loss:
        metrics |= {
            "CrossEntropyLoss": compute_cross_entropy_loss(
                inputs=pred_proba,
                target=target,
                input_score_kind=score_kind,
            ),
        }
        # Hinge loss can only be computed if pred_score is provided
        if pred_score is not None:
            metrics |= {
                "HingeLoss": multiclass_hinge_loss(
                    preds=pred_score,
                    target=target,
                    num_classes=num_classes,
                    squared=False,
                    multiclass_mode="crammer-singer",
                ),
                "SquaredHingeLoss": multiclass_hinge_loss(
                    preds=pred_score,
                    target=target,
                    num_classes=num_classes,
                    squared=True,
                    multiclass_mode="crammer-singer",
                ),
            }

    metrics |= classwise_statistic_metrics(
        preds=preds,
        target=target,
        num_classes=num_classes,
        class_labels=class_labels,
        bootstrap_std_error=bootstrap_std_error,
        metric_names=whitelist,
    )
    metrics |= class_aggregate_statistic_metrics(
        preds=preds,
        target=target,
        num_classes=num_classes,
        bootstrap_std_error=bootstrap_std_error,
        metric_names=whitelist,
    )
    metrics |= roc_metrics(
        pred_proba=pred_proba,
        target=target,
        num_classes=num_classes,
        class_labels=class_labels,
        bootstrap_std_error=bootstrap_std_error,
        metric_names=whitelist,
    )
    metrics |= pr_metrics(
        pred_proba=pred_proba,
        target=target,
        num_classes=num_classes,
        class_labels=class_labels,
        bootstrap_std_error=bootstrap_std_error,
        metric_names=whitelist,
    )
    metrics |= confusion_matrix_metrics(
        preds=preds,
        target=target,
        num_classes=num_classes,
    )

    if top_k_easiest_hardest is not None:
        metrics |= easiest_hardest_examples(
            inputs=pred_proba,
            target=target,
            input_score_kind=score_kind,
            k=top_k_easiest_hardest,
            dataset=dataset,
            id2label=id2label,
        )
    return metrics


def default_metrics(
    preds: Union[np.ndarray, torch.Tensor] = None,
    pred_proba: Union[np.ndarray, torch.Tensor] = None,
    pred_score: Union[np.ndarray, torch.Tensor] = None,
    target: Union[np.ndarray, torch.Tensor] = None,
    score_kind: str = "probabilities",
    include_loss: bool = True,
    num_classes: int = 4,
    class_labels: list = None,
    top_k_easiest_hardest: int | None = None,
    dataset: pd.DataFrame = None,
    id2label: dict[int, str] = None,
    whitelist: list[str] = None,
    bootstrap_std_error: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute loss and metrics, except for TP, FP, TN, FN which can be
    derived from confusion matrix.
    """
    filtered_metrics = {}
    metrics = all_metrics(
        preds=preds,
        pred_proba=pred_proba,
        pred_score=pred_score,
        target=target,
        score_kind=score_kind,
        include_loss=include_loss,
        num_classes=num_classes,
        class_labels=class_labels,
        top_k_easiest_hardest=top_k_easiest_hardest,
        dataset=dataset,
        id2label=id2label,
        whitelist=whitelist,
        bootstrap_std_error=False,
    )
    for k, v in metrics.items():
        if "TP" in k or "FP" in k or "TN" in k or "FN" in k:
            pass
        else:
            filtered_metrics = {**filtered_metrics, k: v}
    return filtered_metrics
