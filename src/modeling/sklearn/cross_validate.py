import numpy as np
import scipy
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
import copy as cp
from typing import Union


def multiclass_cross_validation_predict(
    model: BaseEstimator,
    X: Union[np.ndarray, scipy.sparse.csr.csr_matrix],
    y: np.ndarray,
    k_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 1,
):
    """
    K-fold cross validation for multiclass classification.  This returns the prediction
    results for each fold as well as the model trained for that fold.

    StratifiedKFold is used to maintain balanced distribution for each output class.

    Note: length of each fold may not be equal if original length of dataset is not divisible
        by `k_folds`.  Thus output for each fold is stored in a python array.

    Args:
        model: a sklearn classifier
        X: numpy array or sparse array in CSR format
        y: numpy array of labels
        k_folds: number of K-folds
        shuffle: whether to shuffle each class before splitting into batches
        random_state: seed for shuffling classes

    Returns:
        Array of predicted classes for each fold [np.array(), np.array(), ..., np.array()]
            where each np.array() object is the size of the fold.
        Array of actual classes for each fold [np.array(), np.array(), ..., np.array()]
            where each np.array() object is the size of the fold.
        Array of models trained for each fold [model 1, model 2, ..., model n]
    """
    actual_classes = []
    predicted_classes = []
    models = []

    kf = StratifiedKFold(n_splits=k_folds, shuffle=shuffle, random_state=random_state)

    for train_index, test_index in tqdm(
        kf.split(X, y), total=k_folds, desc=f"{k_folds}-fold Cross Validation"
    ):
        # For each fold, define train and test data
        X_fold_train, X_fold_test = X[train_index], X[test_index]
        y_fold_train, y_fold_test = y[train_index], y[test_index]

        # Ground truth classes for each fold
        actual_classes += [y_fold_test]

        # Train copy of model on each fold
        model_ = cp.deepcopy(model)
        model_.fit(X_fold_train, y_fold_train)
        models += [model_]

        # Get predictions for test set in each fold
        y_fold_pred = model_.predict(X_fold_test)
        predicted_classes += [y_fold_pred]

    return actual_classes, predicted_classes, models