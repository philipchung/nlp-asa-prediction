from __future__ import annotations

import numpy as np
from typing import Union, Any

import shap
import torch
from src.modeling.bert import ClinicalBertModel
from transformers import BertTokenizer, BertTokenizerFast


class ShapClinicalBERTModelWrapper(shap.models.Model):
    """This wraps the model as a SHAP Model Object that can be passed to shap.Explainer."""

    def __init__(
        self,
        pl_model: ClinicalBertModel = None,
        tokenizer: Union[BertTokenizerFast, BertTokenizer] = None,
        device: torch.device = None,
        seq_max_length: int = 512,
        output_score_type: str = "asa",
    ):
        """Initialize model wrapper.

        Args:
            pl_model (ClinicalBertModel): Custom pytorch lightning clinicalbert model.
            tokenizer (Union[BertTokenizerFast, BertTokenizer): Optional. Huggingface Tokenizer.
                Inferred from `pl_model` if not explicitly provided.
            device (torch.device): Optional. Device model and tensor to be placed on.
                Inferred from `pl_model` if not explicitly provided.
            seq_max_length (int): max sequence length for tokenizer to truncate input sequence.
            output_score_type (str): which prediction head's scores to output.
                Valid values are "asa" and "emergency".  Defaults to "asa".
        """
        self.model = pl_model
        self.tokenizer = pl_model.tokenizer if tokenizer is None else tokenizer
        self.device = pl_model.device if device is None else device
        self.seq_max_length = seq_max_length
        assert output_score_type in (
            "asa",
            "emergency",
        ), "Argument `output_score_type` must be 'asa' or 'emergency'"
        self.output_score_type = output_score_type

    def preprocess(
        self, text_list: Union[list[str], list[list[str]]]
    ) -> dict[str, Any]:
        """Take the originally defined inputs, and turn them into something feedable
        to the model.  This uses same tokenizer & preprocessing logic for dataset used
        to train the model."""
        if not isinstance(text_list, list):
            text_list = [text_list]
        # Input text is a list of strings
        tokenized_text = self.tokenizer(
            text_list,
            truncation=True,
            padding="max_length",
            max_length=self.seq_max_length,
        )
        # Tokenizer returns 3 outputs in format of Dict[List[list]].  Convert to tensors.
        input_ids = torch.tensor(tokenized_text["input_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(
            tokenized_text["token_type_ids"], dtype=torch.long
        )
        attention_mask = torch.tensor(
            tokenized_text["attention_mask"], dtype=torch.long
        )

        # Set Dummy Labels (only used for loss calculation)
        # we will ignore loss and only use logits which is the result
        # of a forward pass through each prediction head's linear classification layer
        num_samples_in_batch = input_ids.shape[0]
        asa_labels = torch.zeros(num_samples_in_batch, dtype=torch.long)
        emergency_labels = torch.zeros(num_samples_in_batch, dtype=torch.long)

        model_inputs = {
            "input_ids": input_ids.to(device=self.device),
            "attention_mask": attention_mask.to(device=self.device),
            "token_type_ids": token_type_ids.to(device=self.device),
            "asa_labels": asa_labels.to(device=self.device),
            "emergency_labels": emergency_labels.to(device=self.device),
            "asa_class_weights": self.model.asa_class_weights.to(device=self.device),
            "emergency_class_weights": self.model.emergency_class_weights.to(
                device=self.device
            ),
        }
        return model_inputs

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

        # Preprocess, tokenize string inputs
        model_inputs = self.preprocess(strings)

        # Forward pass through model
        with torch.autocast(device_type=self.model.device.type):
            model_outputs = self.model.model(**model_inputs)

        if self.output_score_type == "asa":
            asa_logits = model_outputs["asa_logits"]
            asa_proba = asa_logits.softmax(dim=1, dtype=torch.float)
            scores = asa_proba
        elif self.output_score_type == "emergency":
            emergency_logits = model_outputs["emergency_logits"]
            emergency_proba = emergency_logits.softmax(dim=1, dtype=torch.float)
            scores = emergency_proba
        return scores.detach().cpu().numpy()
