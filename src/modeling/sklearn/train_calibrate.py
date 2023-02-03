from sklearn.base import BaseEstimator
from pytorch_lightning import LightningDataModule
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd


def train_calibrate_test(
    model: BaseEstimator,
    datamodule: LightningDataModule,
    input_feature_name: str = "input_text",
    output_label_name: str = "asa_label",
    seed=1,
    num_folds=5,
):
    """
    Train calibrated model using stratified k-fold cross validation, then predict on test data.

    This uses CalibratedClassiferCV to jointly perform k-fold cross validation to create
    an ensemble of models.  A regressor is then fit using the ensemble of models to create
    accurate probability predictions for each output class.  Predictions are the output class
    with the highest probability.  Non-calibrated models may generate different results because
    they may not be ensembled and/or their probabilities may not be accurate.

    Args:
        model: sklearn estimator
        datamodule: pytorch lightning datamodule containing train, validation, test splits
            which can be accessed with the methods `train_dataframe()`, `val_dataframe()`,
            `test_dataframe()`.  `prepare_data()` and `setup()` should already have been called
            on datamodule.
        input_feature_name: name of feature containing string of input text for each example
        output_label_name: name of output class to predict
        seed: for classifier and stratified K-Fold cross-validation
        num_folds: number of folds for K-Fold cross-validation

    Return:
        Results dict with keys `model`, `X_train`, `y_train`, `X_test`, `y_test`,
        `y_test_pred`, `y_test_pred_proba`.
    """
    # Define Training & Test Data
    # Since we will use K-fold cross validation, we combine train & validation splits
    train = datamodule.train_dataframe()
    validation = datamodule.val_dataframe()
    test = datamodule.test_dataframe()
    X_train = pd.concat([train[input_feature_name], validation[input_feature_name]])
    y_train = pd.concat([train[output_label_name], validation[output_label_name]])
    X_test = test[input_feature_name]
    y_test = test[output_label_name]

    # Input Data Transform: Text to Unigram + Bigram Counts, then TFIDF transform
    # Note: Liblinear SVM cannot converge with raw counts, must use TFIDF transform
    tfidf_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_test = tfidf_vectorizer.transform(X_test)

    # Label Transform: Convert to numpy matrix (remove original indices from dataset creation)
    y_train = y_train.values
    y_test = y_test.values

    # Train Calibrated Model using Stratified K-Fold Cross-Validation
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cccv = CalibratedClassifierCV(
        base_estimator=model,
        method="sigmoid",
        cv=kf,
        n_jobs=num_folds,
        ensemble=True,
    )
    cccv.fit(X_train, y_train)

    # Generate Predictions on Test Set
    y_test_pred = cccv.predict(X_test)
    y_test_pred_proba = cccv.predict_proba(X_test)

    return {
        "model": cccv,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_test_pred_proba": y_test_pred_proba,
    }
