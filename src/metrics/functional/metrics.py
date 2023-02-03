from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from src.metrics.functional.loss import compute_cross_entropy_loss
from src.metrics.functional.top_examples import easiest_hardest_examples
from src.metrics.metrics import auc_mu
from src.metrics.utils import (
    arraylike_to_tensor,
    unpack_classwise_metrics,
    unpack_statscores,
)
from src.metrics.whitelist import filter_by_whitelist
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


def classwise_statistic_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    class_labels: list = None,
) -> dict[str, torch.Tensor]:
    preds = arraylike_to_tensor(preds)
    target = arraylike_to_tensor(target)
    # Statscores (TP,FP,TN,FN,Support) outputs tensor shape (C,5), with C=num_classes.
    statscores = multiclass_stat_scores(
        preds=preds, target=target, num_classes=num_classes, average=None
    )
    # Convert to {"TP: [class0, class1, ..., classN]", "FP": [...], "TN": [...], "FN": [...]}
    statscores_dict = unpack_statscores(statscores)

    # Each statistic outputs tensor shape (C,), with C=num_classes
    classwise_metrics = {
        **statscores_dict,
        "Precision": multiclass_precision(
            preds=preds, target=target, num_classes=num_classes, average=None
        ),
        "Recall": multiclass_recall(
            preds=preds, target=target, num_classes=num_classes, average=None
        ),
        "F1": multiclass_f1_score(
            preds=preds, target=target, num_classes=num_classes, average=None
        ),
    }
    return unpack_classwise_metrics(classwise_metrics, class_labels)


def class_aggregate_statistic_metrics(
    preds: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
) -> dict[str, torch.Tensor]:
    preds = arraylike_to_tensor(preds)
    target = arraylike_to_tensor(target)
    class_aggregate_metrics = {
        "Precision/Micro": multiclass_precision(
            preds=preds, target=target, num_classes=num_classes, average="micro"
        ),
        "Precision/Macro": multiclass_precision(
            preds=preds, target=target, num_classes=num_classes, average="macro"
        ),
        "Precision/Weighted": multiclass_precision(
            preds=preds, target=target, num_classes=num_classes, average="weighted"
        ),
        "Recall/Micro": multiclass_recall(
            preds=preds, target=target, num_classes=num_classes, average="micro"
        ),
        "Recall/Macro": multiclass_recall(
            preds=preds, target=target, num_classes=num_classes, average="macro"
        ),
        "Recall/Weighted": multiclass_recall(
            preds=preds, target=target, num_classes=num_classes, average="weighted"
        ),
        "F1/Micro": multiclass_f1_score(
            preds=preds, target=target, num_classes=num_classes, average="micro"
        ),
        "F1/Macro": multiclass_f1_score(
            preds=preds, target=target, num_classes=num_classes, average="macro"
        ),
        "F1/Weighted": multiclass_f1_score(
            preds=preds, target=target, num_classes=num_classes, average="weighted"
        ),
        "MCC/MCC": multiclass_matthews_corrcoef(
            preds=preds, target=target, num_classes=num_classes
        ),
        "CohenKappa/Unweighted": multiclass_cohen_kappa(
            preds=preds, target=target, num_classes=num_classes, weights=None
        ),
        "CohenKappa/WeightedLinear": multiclass_cohen_kappa(
            preds=preds, target=target, num_classes=num_classes, weights="linear"
        ),
        "CohenKappa/WeightedQuadratic": multiclass_cohen_kappa(
            preds=preds, target=target, num_classes=num_classes, weights="quadratic"
        ),
    }
    return class_aggregate_metrics


def roc_metrics(
    pred_proba: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    class_labels: list = None,
) -> dict[str, torch.Tensor]:
    pred_proba = arraylike_to_tensor(pred_proba)
    target = arraylike_to_tensor(target)
    roc_classwise_metrics = {
        "AUROC": multiclass_auroc(
            preds=pred_proba,
            target=target,
            num_classes=num_classes,
            average=None,
            thresholds=None,
        ),
    }
    roc_classwise_metrics = unpack_classwise_metrics(
        roc_classwise_metrics, class_labels
    )
    roc_class_aggregate_metrics = {
        "ROC": multiclass_roc(
            preds=pred_proba, target=target, num_classes=num_classes, thresholds=None
        ),
        "AUROC/Macro": multiclass_auroc(
            preds=pred_proba,
            target=target,
            num_classes=num_classes,
            average="macro",
            thresholds=None,
        ),
        "AUROC/Weighted": multiclass_auroc(
            preds=pred_proba,
            target=target,
            num_classes=num_classes,
            average="weighted",
            thresholds=None,
        ),
        "AUROC/AUCmu": torch.tensor(
            auc_mu(y_score=pred_proba.cpu().numpy(), y_true=target.cpu().numpy()),
            dtype=torch.float,
        ),
    }
    roc_metrics = {**roc_classwise_metrics, **roc_class_aggregate_metrics}
    return roc_metrics


def pr_metrics(
    pred_proba: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    num_classes: int = 4,
    class_labels: list = None,
) -> dict[str, torch.Tensor]:
    pred_proba = arraylike_to_tensor(pred_proba)
    target = arraylike_to_tensor(target)
    pr_classwise_metrics = {
        "AUPRC": multiclass_average_precision(
            preds=pred_proba,
            target=target,
            num_classes=num_classes,
            average=None,
            thresholds=None,
        ),
    }
    pr_classwise_metrics = unpack_classwise_metrics(pr_classwise_metrics, class_labels)
    pr_class_aggregate_metrics = {
        "PrecisionRecallCurve": multiclass_precision_recall_curve(
            preds=pred_proba, target=target, num_classes=num_classes, thresholds=None
        ),
        "AUPRC/Macro": multiclass_average_precision(
            preds=pred_proba,
            target=target,
            num_classes=num_classes,
            average="macro",
            thresholds=None,
        ),
        "AUPRC/Weighted": multiclass_average_precision(
            preds=pred_proba,
            target=target,
            num_classes=num_classes,
            average="weighted",
            thresholds=None,
        ),
    }
    pr_metrics = {**pr_classwise_metrics, **pr_class_aggregate_metrics}
    return pr_metrics


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
    top_k_easiest_hardest: int = None,
    dataset: pd.DataFrame = None,
    id2label: dict[int, str] = None,
    whitelist: Union[list[str], dict[str, Any], DictConfig[str, Any]] = None,
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
        whitelist (list, dict, or DictConfig): metrics to output.  All metrics will be
            computed, but metrics not included in whitelist will be dropped.

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

    metrics = {}
    if include_loss:
        loss_metrics = {
            "CrossEntropyLoss": compute_cross_entropy_loss(
                inputs=pred_proba,
                target=target,
                input_score_kind=score_kind,
            ),
        }
        # Hinge loss can only be computed if pred_score is provided
        if pred_score is not None:
            loss_metrics = {
                **loss_metrics,
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
        metrics = {**loss_metrics}

    metrics = {
        **metrics,
        **classwise_statistic_metrics(preds, target, num_classes, class_labels),
        **class_aggregate_statistic_metrics(preds, target, num_classes),
        **roc_metrics(pred_proba, target, num_classes, class_labels),
        **pr_metrics(pred_proba, target, num_classes, class_labels),
        **confusion_matrix_metrics(preds, target, num_classes),
    }
    if top_k_easiest_hardest is not None:
        examples = easiest_hardest_examples(
            inputs=pred_proba,
            target=target,
            input_score_kind=score_kind,
            k=top_k_easiest_hardest,
            dataset=dataset,
            id2label=id2label,
        )
        metrics = {**metrics, **examples}

    # Filter metrics based on whitelist
    if whitelist:
        metrics = filter_by_whitelist(
            metrics=metrics, class_labels=class_labels, whitelist=whitelist
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
    top_k_easiest_hardest: int = None,
    dataset: pd.DataFrame = None,
    id2label: dict[int, str] = None,
    whitelist: list[str] = None,
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
    )
    for k, v in metrics.items():
        if "TP" in k or "FP" in k or "TN" in k or "FN" in k:
            pass
        else:
            filtered_metrics = {**filtered_metrics, k: v}
    return filtered_metrics
