import numpy as np
import pandas as pd
from pytorch_lightning import LightningDataModule

import fasttext


def evaluate_fasttext(
    model: fasttext.FastText._FastText,
    datamodule: LightningDataModule,
    split: str = "validation",
    input_feature_name: str = "input_text",
    output_label_name: str = "asa_label",
    num_output_classes: int = 4,
):
    """
    Evaluate a fasttext model.
    Args:
        model (fasttext model): the model to evaluate
        datamodule (pytorch lightning datamodule): Must contain method `export_fasttext()`
            which exports training data into a text file that fasttext uses for model training.
            `prepare_data()` and `setup()` should already have been called
            on datamodule.
        split (str): dataset split to evaluate model on {"train", "validation", "test", "predict"}
        input_feature_name (str): name of feature containing input text for each example
        output_label_name (str): name of output class to predict
        num_output_classes (int): number of output classes to predict
    Returns:
        Dictionary of predictions for split.
        {
            "preds": [3, 1, 2, ...],
            "target": [3, 2, 2, ...],
            "pred_proba": [[0.1, 0.2, 0.7], [0.7, 0.1, 0.2] ....]
        }
    """
    # Get data for split
    fn = datamodule.get_data_getter(split=split, getter_type="dataframe")
    data_df = fn(columns=[input_feature_name, output_label_name])
    output_label_data_type = data_df[output_label_name].dtype
    # Evaluate Model
    df = (
        data_df[input_feature_name]
        .apply(lambda example: model.predict(example, k=num_output_classes))
        .apply(parse_fasttext_prediction)
        .apply(pd.Series)
    )
    df["preds"] = df["preds"].astype(output_label_data_type)
    results = pd.concat(
        [data_df[output_label_name].rename("target"), df], axis=1
    ).to_dict(orient="list")
    return {k: np.array(v) for k, v in results.items()}


def parse_fasttext_prediction(result: tuple, label_prefix: str = "__label__"):
    """
    Format FastText output to top class prediction & probability for each class.
    By default, fasttext's `model.predict(input_text)` will only output the top prediction.
    Specifying arguments `model.predict(input_text, k=n)` will give the top n predictions
    and the associated probabilities as a tuple.  These predictions are sorted by highest
    probability to lowest.  This method will transform this tuple into a dict with
    output probabilities stored in an array in order of class labels.
    Args:
        result (tuple): tuple(tuple("__label__2", "__label__0, "__label__1"), array([prob_2, prob_0, prob_1]))
    Returns:
        Dictionary with reformatted results that conform with sklearn nomenclature:
        {
            "preds": "2",
            "pred_proba": [prob_0, prob_1, prob_2]
        }
    """
    predicted_results = [x.lstrip(label_prefix) for x in result[0]]
    predicted_probability = result[1]
    predictions = list(zip(predicted_results, predicted_probability))
    predictions = sorted(predictions, key=lambda x: x[0])
    predicted_proba = [x[1] for x in predictions]
    return {
        "preds": predicted_results[0],
        "pred_proba": predicted_proba,
    }
