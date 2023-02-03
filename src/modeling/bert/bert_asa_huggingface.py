from dataclasses import dataclass
from transformers import BertPreTrainedModel, BertModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from typing import Optional, Tuple


class BertForSequenceClassificationForASAPrediction(BertPreTrainedModel):
    """
    Dual Prediction Heads for ASA Class & Emergent Classifier Prediction.

    Combined loss is a blend of the loss from both heads.  The contribution from
    each head to the final loss can be adjusted/tuned as a hyperparameter.

    Loss function is cross entropy loss with label smoothing.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.asa_head = BertOnlyASAHead(config)
        self.emergency_head = BertOnlyEmergencyHead(config)

        self.asa_classifier_head_weight = config.asa_classifier_head_weight
        self.emergency_classifier_head_weight = config.emergency_classifier_head_weight

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        asa_labels=None,
        emergency_labels=None,
        asa_class_weights=None,
        emergency_class_weights=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # Compute Loss for ASA Class
        asa_outputs = self.asa_head(pooled_output, asa_labels, weight=asa_class_weights)
        asa_logits = asa_outputs["asa_logits"]
        asa_loss = asa_outputs["asa_loss"]

        # Compute Loss for Emergency Modifier
        emergency_outputs = self.emergency_head(
            pooled_output, emergency_labels, weight=emergency_class_weights
        )
        emergency_logits = emergency_outputs["emergency_logits"]
        emergency_loss = emergency_outputs["emergency_loss"]

        # Combined Weighted-Average Loss
        combined_loss = (
            asa_loss * self.asa_classifier_head_weight
            + emergency_loss * self.emergency_classifier_head_weight
        ) / (self.asa_classifier_head_weight + self.emergency_classifier_head_weight)

        return SequenceClassifierOutputForASAPrediction(
            combined_loss=combined_loss,
            asa_loss=asa_loss,
            emergency_loss=emergency_loss,
            asa_logits=asa_logits,
            emergency_logits=emergency_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertOnlyASAHead(nn.Module):
    "Compute Loss for ASA Prediction."

    def __init__(self, config):
        super().__init__()
        self.asa_num_labels = config.asa_num_labels
        self.label_smoothing_alpha = config.label_smoothing_alpha
        self.asa_classifier = nn.Linear(config.hidden_size, self.asa_num_labels)

    def forward(
        self,
        pooled_output: torch.Tensor = None,
        labels: torch.Tensor = None,
        weight: torch.Tensor = None,
        reduction: str = "mean",
    ):
        "Must provide pooled_output OR logits."
        logits = self.asa_classifier(pooled_output)
        loss_fn = CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.label_smoothing_alpha,
            reduction=reduction,
        )
        loss = loss_fn(logits.view(-1, self.asa_num_labels), labels)
        return {
            "asa_loss": loss,
            "asa_logits": logits,
        }


class BertOnlyEmergencyHead(nn.Module):
    "Compute Loss for Emergency Prediction."

    def __init__(self, config):
        super().__init__()
        self.emergency_num_labels = config.emergency_num_labels
        self.label_smoothing_alpha = config.label_smoothing_alpha
        self.emergency_classifier = nn.Linear(
            config.hidden_size, self.emergency_num_labels
        )

    def forward(
        self,
        pooled_output: torch.Tensor = None,
        labels: torch.Tensor = None,
        weight: torch.Tensor = None,
        reduction: str = "mean",
    ):
        logits = self.emergency_classifier(pooled_output)
        loss_fn = CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.label_smoothing_alpha,
            reduction=reduction,
        )
        loss = loss_fn(logits.view(-1, self.emergency_num_labels), labels)
        return {"emergency_loss": loss, "emergency_logits": logits}


@dataclass
class SequenceClassifierOutputForASAPrediction(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        combined_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss combined across all prediction heads.
        asa_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss for ASA class prediction head.
        emergency_loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification loss for emergency class prediciton head.
        asa_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax) for ASA class prediction head.
        emergency_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax) for emergency class prediction head.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or
            when `config.output_hidden_states=True`): Tuple of `torch.FloatTensor` (one for the output of the
            embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when
            `config.output_attentions=True`): Tuple of `torch.FloatTensor` (one for each layer) of shape
            `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the attention softmax,
            used to compute the weighted average in the self-attention heads.
    """

    combined_loss: Optional[torch.FloatTensor] = None
    asa_loss: Optional[torch.FloatTensor] = None
    emergency_loss: Optional[torch.FloatTensor] = None
    asa_logits: torch.FloatTensor = None
    emergency_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertConfigForASAPrediction(PretrainedConfig):
    r"""
    Same as BertConfig from Huggingface Transformers library with the addition of
    the following arguments:
        asa_num_labels (int): number of labels in ASA classification
        emergency_num_labels (int): number of labels in emergency classification
        asa_classifier_head_weight (float, defaults to 1.0): Weighing factor
            when combining loss from ASA classifier head.
        emergency_classifier_head_weight (float, defaults to 1.0): Weighing factor
            when combining loss from emergency classifier head.
        label_smoothing_alpha (float, defaults to 0.0): Value from 0.0-1.0.
            Cntrols degree of label smoothing applied to cross-entropy loss function
    ```"""
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        asa_num_labels=None,
        emergency_num_labels=None,
        asa_classifier_head_weight=1.0,
        emergency_classifier_head_weight=1.0,
        label_smoothing_alpha=0.0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.asa_num_labels = asa_num_labels
        self.emergency_num_labels = emergency_num_labels
        self.asa_classifier_head_weight = asa_classifier_head_weight
        self.emergency_classifier_head_weight = emergency_classifier_head_weight
        self.label_smoothing_alpha = label_smoothing_alpha
