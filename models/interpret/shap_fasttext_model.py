from __future__ import annotations

import numpy as np
from typing import Union

import shap
import fasttext
from src.modeling.fasttext.evaluate import parse_fasttext_prediction


class ShapFastTextModelWrapper(shap.models.Model):
    """This wraps the model as a SHAP Model Object that can be passed to shap.Explainer."""

    def __init__(
        self,
        fasttext_model: fasttext.FastText._FastText = None,
        label2id: dict = {"I": 0, "II": 1, "III": 2, "IV-V": 3},
        label_prefix: str = "__label__",
    ):
        """Initialize model wrapper.

        Args:
            fasttext_model (fasttext.FastText._FastText): Fasttext model.
            label2id (dict): mapping between class string label and integer id
            label_prefix (str): prefix to integer id used in fasttext label format
        """
        self.model = fasttext_model
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.label_prefix = label_prefix
        self.fasttextlabel2id = {
            f"{label_prefix}{integer_id}": integer_id
            for integer_id in self.id2label.keys()
        }
        self.id2fasttextlabel = {v: k for k, v in self.fasttextlabel2id.items()}

    def __call__(self, strings: Union[str, list[str], np.array]) -> np.array:
        """Transform model input strings into output predictions.

        Args:
            strings (Union[str, list[str], np.array]): List of input strings, or
                numpy array of input strings.  If plain string is provided, it will
                be wrapped in a list.

        Returns:
            np.array: numpy array of output predictions with dimension (N, C)
                where N is number of input string examples and C is number of output
                classes.
        """
        if isinstance(strings, str):
            strings = [strings]
        elif isinstance(strings, np.ndarray):
            # Shap Model Masking calls with numpy array of strings, which needs to be
            # converted to standard list of strings for tokenizer to process
            strings = list(strings)

        # Get output prediction probability scores from model for each string
        outputs = [self.model.predict(text=sentence, k=4) for sentence in strings]
        parsed_outputs = [
            parse_fasttext_prediction(output, label_prefix=self.label_prefix)
            for output in outputs
        ]
        scores = [output["pred_proba"] for output in parsed_outputs]
        return np.array(scores)
