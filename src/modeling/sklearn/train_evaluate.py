from sklearn.base import BaseEstimator
from pytorch_lightning import LightningDataModule
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from typing import Union


def train_sklearn(
    model: BaseEstimator,
    datamodule: LightningDataModule,
    input_feature_name: str = "input_text",
    output_label_name: str = "asa_label",
):
    """
    Train a sklearn model from train dataset in datamodule using unigram-bigram + TFIDF
    text feature transformation.  This method creates a sklearn Pipeline that packages
    the feature transformation along with the model.

    Args:
        model: sklearn estimator
        datamodule: pytorch lightning datamodule containing train, validation, test splits
            which can be accessed with the methods `train_dataframe()`, `val_dataframe()`,
            `test_dataframe()`.  `prepare_data()` and `setup()` should already have been called
            on datamodule.
        input_feature_name: name of feature containing string of input text for each example
        output_label_name: name of output class to predict

    Return:
        Results dict with keys `pipeline`, `X_train`, `y_train`.
    """
    # Define Training & Test Data
    train = datamodule.train_dataframe()
    X_train = train[input_feature_name]
    y_train = train[output_label_name]

    # Input Data Transform: Text to Unigram + Bigram Counts, then TFIDF transform
    # Note: Liblinear SVM cannot converge with raw counts, must use TFIDF transform
    tfidf_vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))

    # Create Pipeline
    pipeline = Pipeline([("TFIDF", tfidf_vectorizer), ("estimator", model)])

    # Label Transform: Convert to numpy matrix (remove original indices from dataset creation)
    y_train = y_train.values

    # Train Model
    pipeline.fit(X_train, y_train)

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "y_train": y_train,
    }


def evaluate_sklearn(
    pipeline: Union[Pipeline, BaseEstimator],
    datamodule: LightningDataModule,
    split: str = "validation",
    input_feature_name: str = "input_text",
    output_label_name: str = "asa_label",
    model_type: str = None,
):
    """
    Evaluate a sklearn model using data split in datamodule.

    If model is a support vector machine classifier, will have attribute `decision_function`
    which is the raw score for each example.  These will be included in the results dict
    as "y_eval_pred_score".

    Args:
        pipeline: sklearn pipeline or estimator
        datamodule: pytorch lightning datamodule containing train, validation, test splits
            which can be accessed with the methods `train_dataframe()`, `val_dataframe()`,
            `test_dataframe()`.  `prepare_data()` and `setup()` should already have been called
            on datamodule.
        split: data split to use for evaluation {"train", "validation", "test"}
        input_feature_name: name of feature containing string of input text for each example
        output_label_name: name of output class to predict
        model_type: "svm" for Support Vector Machine Classifier, "rf" Random Forest, or `None`.
            svm has `decision_function` method to compute scores for each output class.  These
            are unnormalized confidence scores on which the final prediction is made, but are
            not probabilities. These probabilities are included in the return dict as
            `y_eval_pred_score`.
            Random Forest has `pred_proba` method which computes probability for for each
            output class.  The final prediction is made based on these probabilities.  These
            probabilities are included in the return dict as `y_eval_pred_proba`.
            If `None` is selected, the return dictionary will include `y_eval_pred` which
            is the predicted class, but no other prediction outputs will be given.

    Return:
        Results dict with keys `pipeline`, `X_eval`, `y_eval`, `y_eval_pred`,
        `y_eval_pred_proba`, `y_eval_pred_score`.
    """
    # Get the Evaluation Data Split
    data_getter_fn = datamodule.get_data_getter(split=split, getter_type="dataframe")
    eval_set = data_getter_fn()
    X_eval = eval_set[input_feature_name]
    y_eval = eval_set[output_label_name]

    # Label Transform: Convert to numpy matrix (remove original indices from dataset creation)
    y_eval = y_eval.values

    # Generate Predictions
    y_eval_pred = pipeline.predict(X_eval)
    y_eval_pred_proba = pipeline.predict_proba(X_eval) if model_type == "rf" else None
    y_eval_pred_score = (
        pipeline.decision_function(X_eval) if model_type == "svm" else None
    )

    return {
        "pipeline": pipeline,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "y_eval_pred": y_eval_pred,
        "y_eval_pred_proba": y_eval_pred_proba,
        "y_eval_pred_score": y_eval_pred_score,
    }


def validate_sklearn(
    pipeline: Union[Pipeline, BaseEstimator],
    datamodule: LightningDataModule,
    input_feature_name: str = "input_text",
    output_label_name: str = "asa_label",
    model_type: str = None,
):
    """
    Evaluate sklearn model on validation dataset in datamodule.

    Args:
        See arguments for `evaluate_sklearn()`.

    Return:
        Results dict with keys `pipeline`, `X_val`, `y_val`, `y_val_pred`,
        `y_val_pred_proba`, `y_val_pred_score`.
    """
    outputs = evaluate_sklearn(
        pipeline=pipeline,
        datamodule=datamodule,
        split="validation",
        input_feature_name=input_feature_name,
        output_label_name=output_label_name,
        model_type=model_type,
    )
    return {
        "pipeline": outputs["pipeline"],
        "X_val": outputs["X_eval"],
        "y_val": outputs["y_eval"],
        "y_val_pred": outputs["y_eval_pred"],
        "y_val_pred_proba": outputs["y_eval_pred_proba"],
        "y_val_pred_score": outputs["y_eval_pred_score"],
    }


def test_sklearn(
    pipeline: Union[Pipeline, BaseEstimator],
    datamodule: LightningDataModule,
    input_feature_name: str = "input_text",
    output_label_name: str = "asa_label",
    model_type: str = None,
):
    """
    Evaluate sklearn moodel on test dataset in datamodule.

    Args:
        See arguments for `evaluate_sklearn()`.

    Return:
        Results dict with keys `pipeline`, `X_test`, `y_test`, `y_test_pred`,
        `y_test_pred_proba`, `y_test_pred_score`.
    """
    outputs = evaluate_sklearn(
        pipeline=pipeline,
        datamodule=datamodule,
        split="test",
        input_feature_name=input_feature_name,
        output_label_name=output_label_name,
        model_type=model_type,
    )
    return {
        "pipeline": outputs["pipeline"],
        "X_test": outputs["X_eval"],
        "y_test": outputs["y_eval"],
        "y_test_pred": outputs["y_eval_pred"],
        "y_test_pred_proba": outputs["y_eval_pred_proba"],
        "y_test_pred_score": outputs["y_eval_pred_score"],
    }


def train_validate_sklearn(
    model: BaseEstimator,
    datamodule: LightningDataModule,
    input_feature_name: str = "input_text",
    output_label_name: str = "asa_label",
    model_type: str = None,
):
    """
    Train a sklearn model from train dataset in datamodule using unigram-bigram + TFIDF
    text feature transformation, then evaluate it on validation dataset.

    Args:
        pipeline: sklearn pipeline or estimator
        datamodule: pytorch lightning datamodule containing train, validation, test splits
            which can be accessed with the methods `train_dataframe()`, `val_dataframe()`,
            `test_dataframe()`.  `prepare_data()` and `setup()` should already have been called
            on datamodule.
        input_feature_name: name of feature containing string of input text for each example
        output_label_name: name of output class to predict
        model_type: model-specific probability or score calculation {"svm", "rf", None}

    Return:
        Results dict with keys `pipeline`, `X_train`, `y_train`, `X_val`,
        `y_val`, `y_val_pred`, `y_val_pred_proba`, `y_val_pred_score`.
    """
    train_outputs = train_sklearn(
        model, datamodule, input_feature_name, output_label_name
    )
    pipeline = train_outputs.pop("pipeline")
    validation_outputs = validate_sklearn(
        pipeline=pipeline,
        datamodule=datamodule,
        input_feature_name=input_feature_name,
        output_label_name=output_label_name,
        model_type=model_type,
    )
    return {**train_outputs, **validation_outputs}
