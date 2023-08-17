# Source: https://github.com/kleimanr/auc_mu/blob/master/auc_mu.py
# Modified to produce metric output with warning when not all labels are represented
# in data passed into y_true.  For example, if we have n_classes = 4 with labels
# [0, 1, 2, 3], but data only has labels [1, 2, 3] and is missing label 0, we
# will still get metric output.
"""
Computation of the measure 'AUC Mu'. This measure requires installation of the
numpy and sklearn libraries.

This code corresponds to the paper: Kleiman, R., Page, D. ``AUC Mu: A
Performance Metric for Multi-Class Machine Learning Models``, Proceedings of the
2019 International Conference on Machine Learning (ICML).
"""

__author__ = "Ross Kleiman"
__copyright__ = "Copyright 2019"
__credits__ = ["Ross Kleiman"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Ross Kleiman"
__email__ = "rkleiman@cs.wisc.edu"
__status__ = "Production"

import warnings
from typing import Union

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def multiclass_aucmu(
    preds: Union[torch.tensor, np.ndarray], target: Union[torch.tensor, np.ndarray], **kwargs
) -> float:
    "Wrapper for `auc_mu` so arguments follow the `preds` and `target` convention."
    y_score = preds.cpu().numpy()
    y_true = target.cpu().numpy()
    auc_mu_value = auc_mu(y_true=y_true, y_score=y_score, **kwargs)
    return torch.tensor(auc_mu_value, dtype=torch.float)


# ----------------------------------------------------------------------
def auc_mu(y_true, y_score, A=None, W=None) -> float:
    """
    Compute the multi-class measure AUC Mu from prediction scores and labels.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        The true class labels in the range [0, n_samples-1]

    y_score : array, shape = [n_samples, n_classes]
        Target scores, where each row is a categorical distribution over the
        n_classes.

    A : array, shape = [n_classes, n_classes], optional
        The partition (or misclassification cost) matrix. If ``None`` A is the
        argmax partition matrix. Entry A_{i,j} is the cost of classifying an
        instance as class i when the true class is j. It is expected that
        diagonal entries in A are zero and off-diagonal entries are positive.

    W : array, shape = [n_classes, n_classes], optional
        The weight matrix for incorporating class skew into AUC Mu. If ``None``,
        the standard AUC Mu is calculated. If W is specified, it is expected to
        be a lower triangular matrix where entrix W_{i,j} is a positive float
        from 0 to 1 for the partial score between classes i and j. Entries not
        in the lower triangular portion of W must be 0 and the sum of all
        entries in W must be 1.

    Returns
    -------
    auc_mu : float

    References
    ----------
    .. [1] Kleiman, R., Page, D. ``AUC Mu: A Performance Metric for Multi-Class
           Machine Learning Models``, Proceedings of the 2019 International
           Conference on Machine Learning (ICML).

    """

    # Validate input arguments
    if not isinstance(y_score, np.ndarray):
        raise TypeError("Expected y_score to be np.ndarray, got: %s" % type(y_score))
    if not y_score.ndim == 2:
        raise ValueError("Expected y_score to be 2 dimensional, got: %s" % y_score.ndim)
    n_samples, n_classes = y_score.shape

    if not isinstance(y_true, np.ndarray):
        raise TypeError("Expected y_true to be np.ndarray, got: %s" % type(y_true))
    if not y_true.ndim == 1:
        raise ValueError("Expected y_true to be 1 dimensional, got: %s" % y_true.ndim)
    if not y_true.shape[0] == n_samples:
        raise ValueError(
            "Expected y_true to be shape %s, got: %s" % (str(y_score.shape), str(y_true.shape))
        )
    unique_labels = np.unique(y_true)
    if not np.all(unique_labels == np.arange(n_classes)):
        not_represented = list(set(range(n_classes)) - set(unique_labels))
        warnings.warn(
            f"Samples do not contain label for class(es) {not_represented}.  "
            "AUC-mu calculation assumes separability of classes from non-represented classes "
            "to be zero.  If this is not accurate, then the AUC-mu value is inaccurate "
            "and should not be trusted."
        )

    if A is None:
        A = np.ones((n_classes, n_classes)) - np.eye(n_classes)
    if not isinstance(A, np.ndarray):
        raise TypeError("Expected A to be np.ndarray, got: %s" % type(A))
    if not A.ndim == 2:
        raise ValueError("Expected A to be 2 dimensional, got: %s" % A.ndim)
    if not A.shape == (n_classes, n_classes):
        raise ValueError(
            "Expected A to be shape (%i, %i), got: %s" % (n_classes, n_classes, str(A.shape))
        )
    if not np.all(A.diagonal() == np.zeros(n_classes)):
        raise ValueError("Expected A to be zero on the diagonals")
    if not np.all(A >= 0):
        raise ValueError("Expected A to be non-negative")

    if W is None:
        W = np.tri(n_classes, k=-1)
        W /= W.sum()
    if not isinstance(W, np.ndarray):
        raise TypeError("Expected W to be np.ndarray, got: %s" % type(W))
    if not W.ndim == 2:
        raise ValueError("Expected W to be 2 dimensional, got: %s" % W.ndim)
    if not W.shape == (n_classes, n_classes):
        raise ValueError(
            "Expected W to be shape (%i, %i), got: %s" % (n_classes, n_classes, str(W.shape))
        )

    auc_total = 0.0

    for class_i in range(n_classes):
        preds_i = y_score[y_true == class_i]
        n_i = preds_i.shape[0]
        for class_j in range(class_i):
            preds_j = y_score[y_true == class_j]
            temp_preds = np.vstack((preds_i, preds_j))
            n_j = preds_j.shape[0]
            n = n_i + n_j

            temp_labels = np.zeros((n), dtype=int)
            temp_labels[n_i:n] = 1

            v = A[class_i, :] - A[class_j, :]
            scores = np.dot(temp_preds, v)

            try:
                # roc_auc_score can be computed when both pos & neg labels appear in temp_labels
                score_i_j = roc_auc_score(temp_labels, scores)
            except:
                # if only one class of labels appears, cannot compute roc_auc.
                # set score to 0.  Score_i_j is a separability measurement
                # between class i & class j.  If data does not have labels of one class
                # we cannot compute the separability of the 2 classes.  Thus we set
                # score to 0 to indicate we have no knowledge of separability.
                # If this is a false assumption, then the AUC-mu we compute is inaccurate.
                score_i_j = 0
            auc_total += W[class_i, class_j] * score_i_j

    return auc_total
