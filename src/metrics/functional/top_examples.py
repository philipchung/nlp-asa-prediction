from __future__ import annotations

import warnings
from typing import Any, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from src.metrics.functional.loss import compute_loss_per_example
from src.metrics.utils import arraylike_to_tensor


def top_k_loss(
    inputs: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    input_score_kind: str = "probabilities",
    kind: str = "hardest",
    k: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply loss function to each example to find k easiest/hardest examples in dataset.
    If `inputs` are logits, set `input_score_kind`="logits" to use CrossEntropyLoss.
    If `inputs` are probabilities, set `input_score_kind`="probabilities to perform
        log transform and then negative log likelihood loss.

    Args:
        inputs (numpy array or torch tensor): output probabilities or logit scores
            for each class for each example.  Shape (N,C) where N is number of examples,
            C is number of classes.
        target (numpy array or torch tensor): target class for each example.
            Shape is (N) where N is number of examples.
        input_score_kind (str, torch.nn loss module): either {"logits" or "probabilities},
            which affects the loss function used to find easiest/hardest examples.
        kind (str): type of ranking to apply {"easiest", "hardest"}
        k (int): how many examples to return.  If `None`, returns all examples.

    Returns:
        Losses for top k easiest/hardest examples.
        Indicies in inputs that correspond to those examples.

    """
    losses = compute_loss_per_example(inputs, target, input_score_kind)

    if kind.lower() == "easiest":
        # Get indices for examples in order with lowest loss first
        sorted_losses = torch.sort(losses, descending=False, stable=True)
    elif kind.lower() == "hardest":
        # Get indices for examples in order with highest loss first
        sorted_losses = torch.sort(losses, descending=True, stable=True)
    else:
        raise ValueError("Invalid `kind` argument. Must be `easiest` or `hardest`.")
    values, indices = sorted_losses

    if k is None:
        return values, indices
    else:
        return values[:k], indices[:k]


def top_k_examples(
    inputs: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    input_score_kind: str = "probabilities",
    kind: str = "hardest",
    k: int = 10,
    dataset: Union[Dataset, pd.DataFrame] = None,
    id2label: dict[int, str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Get top k easiest or hardest examples from dataset.

    Order of examples must be preserved from inputs & dataset, otherwise
    random examples will be returned.
    (e.g. item j in inputs must correspond to same example as item j in dataset)
    """
    if id2label is None:
        warnings.warn(
            "Missing argument to `id2label`.  Unable to generate label names for examples."
        )

    inputs = arraylike_to_tensor(inputs)
    target = arraylike_to_tensor(target)
    # Get actual predictions from input probabilities
    predicted_labels = torch.argmax(inputs, dim=1)
    # Get top k loss & indicies for corresponding inputs
    losses, indices = top_k_loss(inputs, target, input_score_kind, kind, k)

    # Format top k inputs, predictions, labels, losses into table
    top_k_predicted_labels = predicted_labels[indices].cpu().numpy()
    top_k_predicted_label_names = (
        [id2label[x] for x in top_k_predicted_labels] if id2label else None
    )
    top_k_inputs = inputs[indices].cpu().numpy().round(decimals=2)
    top_k_losses = losses.cpu().numpy().round(decimals=2)
    cols_to_concat = (
        pd.Series(top_k_predicted_labels, name="predicted_label"),
        pd.Series(top_k_predicted_label_names, name="predicted_label_name"),
        pd.Series(top_k_inputs.tolist(), name=input_score_kind.capitalize()),
        pd.Series(top_k_losses, name="Loss"),
    )
    df_to_concat = pd.concat([x for x in cols_to_concat if x is not None], axis=1)

    # Get corresponding top k examples from dataset and concatenate with table
    # NOTE: if data loaded via torch DataLoader and shuffled, we will not
    # get the correct examples here--we must get original indices on the dataset.
    if isinstance(dataset, Dataset):
        examples = dataset.select(indices).to_pandas().set_index("index")
    elif isinstance(dataset, pd.DataFrame):
        examples = dataset.iloc[indices, :]
    else:
        raise ValueError("Unknown argument for `dataset`.")
    df = pd.concat([examples.reset_index(drop=False), df_to_concat], axis=1).set_index(
        "index"
    )
    return {f"Examples/{kind.capitalize()}": df}


def easiest_hardest_examples(
    inputs: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    input_score_kind: str = "probabilities",
    k: int = 10,
    dataset: Dataset = None,
    id2label: dict[int, str] = None,
) -> dict[str, Any]:
    "Get easiest and hardest examples from dataset."
    hardest = top_k_examples(
        inputs, target, input_score_kind, "hardest", k, dataset, id2label
    )
    easiest = top_k_examples(
        inputs, target, input_score_kind, "easiest", k, dataset, id2label
    )
    return {**hardest, **easiest}
