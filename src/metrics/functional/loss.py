import torch
from torch.nn import CrossEntropyLoss, NLLLoss
import numpy as np
from src.metrics.utils import arraylike_to_tensor
from typing import Union
import warnings


def compute_cross_entropy_loss(
    inputs: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    input_score_kind: str = "probabilities",
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute CrossEntropy loss from logits or probabilities, depending on `input_score_kind`.
    If `inputs` are logits, set `input_score_kind`="logits" to use CrossEntropyLoss.  Logits
        are raw scores output by a model and can be any real number (-inf, +inf).
    If `inputs` are probabilities, set `input_score_kind`="probabilities to perform
        log transform and then negative log likelihood loss.  Probabilities are bounded
        from [0,1].

    By default, this method averages loss over all examples.  If loss for each example
    is desired, set `reduction`="none".

    Args:
        inputs (numpy array or torch tensor): output probabilities or logit scores
            for each class for each example.  Shape (N,C) where N is number of examples,
            C is number of classes.
        target (numpy array or torch tensor): target class for each example.
            Shape is (N) where N is number of examples.
        input_score_kind (str, torch.nn loss module): either {"logits", "probabilities},
            which affects the loss function used to find easiest/hardest examples.
        reduction (str): type of reduction to apply in loss function {"mean", "none"}

    Returns:
        Indicies in dataset that is the top k easiest/hardest examples.
        Losses for those corresponding examples.

    """
    inputs = arraylike_to_tensor(inputs)
    target = arraylike_to_tensor(target)

    if input_score_kind == "probabilities":
        # inputs are probabilities (e.g. already "softmax activated"), so compute Neg-Log Likelihood
        eps = 1e-10
        log_proba = torch.log(inputs + eps)
        loss_fn = NLLLoss(reduction=reduction)
        losses = loss_fn(log_proba, target)
        if losses.isnan().any():
            warnings.warn(
                "Loss contains `nan` values.  This commonly occurs if the "
                "wrong `input_score_kind` is specificed (i.e. if `inputs` contains logits but you "
                "specified `probabilities` instead of `logits`."
            )
    elif input_score_kind == "logits":
        # inputs are scores that are not "softmax activated", so compute CrossEntropyLoss
        # CrossEntropyLoss(x) = NLLLoss(log(softmax(x)))
        loss_fn = CrossEntropyLoss(reduction=reduction)
        losses = loss_fn(inputs, target)
    else:
        raise ValueError("Unknown value for argument `input_score_kind`.")
    return losses


def compute_loss_per_example(
    inputs: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    input_score_kind: str = "probabilities",
) -> torch.Tensor:
    "Get raw loss scores for each example (no aggregation)."
    return compute_cross_entropy_loss(
        inputs=inputs, target=target, input_score_kind=input_score_kind, reduction="none"
    )
