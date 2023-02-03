from __future__ import annotations

from typing import Optional, Union, Any

import pytorch_lightning as pl
import torch
from src.modeling.utils import detach
from transformers import AdamW as TransformersAdamW
from transformers import BertTokenizer, BertTokenizerFast, BertConfig

from . import BertConfigForASAPrediction, BertForSequenceClassificationForASAPrediction


class ClinicalBertModel(pl.LightningModule):
    """
    Pytorch Lightning Module wrapped around a Huggingface ClinicalBERT model.
    """

    def __init__(
        self,
        precision: int = 32,
        pretrained_model: str = None,
        bert_config: Optional[BertConfigForASAPrediction] = None,
        tokenizer: Union[BertTokenizerFast, BertTokenizer] = None,
        asa_label2id: dict = None,
        emergency_label2id: dict = None,
        asa_class_weights: torch.Tensor = None,
        emergency_class_weights: torch.Tensor = None,
        asa_classifier_head_weight: float = 1.0,
        emergency_classifier_head_weight: float = 1.0,
        epochs: int = None,
        batch_size: int = None,
        accumulate_grad_batches: int = None,
        optimizer: Union[str, torch.optim.Optimizer] = None,
        learning_rate: float = None,
        weight_decay: float = 0,
        adam_eps: float = 1e-6,
        learning_rate_schedule: str = None,
        dropout: float = None,
        label_smoothing_alpha: float = 0,
        *args,
        **kwargs,
    ):
        "Define model parameters."
        super().__init__()
        self.save_hyperparameters()
        # Model Initialization Properties
        self.precision = precision
        dtype_map = {
            16: torch.float16,
            32: torch.float32,
            64: torch.float64,
            "bf16": torch.bfloat16,
        }
        self.torch_dtype = dtype_map[precision]
        self.pretrained_model = pretrained_model
        self.bert_config = bert_config
        self.tokenizer = (
            tokenizer
            if tokenizer
            else BertTokenizerFast.from_pretrained(pretrained_model)
        )
        self.asa_label2id = asa_label2id
        self.asa_id2label = {v: k for k, v in asa_label2id.items()}
        self.emergency_label2id = emergency_label2id
        self.emergency_id2label = {v: k for k, v in emergency_label2id.items()}
        self.asa_class_weights = asa_class_weights.to(dtype=self.torch_dtype)
        self.emergency_class_weights = emergency_class_weights.to(
            dtype=self.torch_dtype
        )
        self.asa_num_labels = len(asa_label2id)
        self.emergency_num_labels = len(emergency_label2id)
        self.asa_classifier_head_weight = asa_classifier_head_weight
        self.emergency_classifier_head_weight = emergency_classifier_head_weight

        # Model Training Settings
        self.epochs = epochs  # Num epochs to train
        self.batch_size = batch_size
        self.accumulate_grad_batches = accumulate_grad_batches
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.learning_rate_schedule = learning_rate_schedule
        self.dropout = dropout
        self.label_smoothing_alpha = label_smoothing_alpha

        # Create BERT Model
        self.get_bert_config(bert_config)
        self.configure_model()

        # Set label2id & id2label with ASA class mapping
        self.model.config.label2id = self.asa_label2id
        self.model.config.id2label = self.asa_id2label

    def get_bert_config(self, bert_config: BertConfig) -> None:
        "Create Transformers Bert Config."
        if bert_config:
            self.bert_config = bert_config
        else:
            bert_config = BertConfigForASAPrediction.from_pretrained(
                self.pretrained_model
            )
            bert_config.update(
                {
                    ## Model Architecture Configuration
                    # vocab_size=30522,
                    # hidden_size=768,
                    # num_hidden_layers=12,
                    # num_attention_heads=12,
                    # intermediate_size=3072,
                    # hidden_act="gelu",
                    "hidden_dropout_prob": self.dropout,
                    "attention_probs_dropout_prob": self.dropout,
                    "classifier_dropout": self.dropout,
                    "max_position_embeddings": 512,
                    # initializer_range=0.02,
                    # layer_norm_eps=1e-12,
                    # pad_token_id=0,
                    # gradient_checkpointing=False,
                    # position_embedding_type="absolute",
                    ### Fine-tuning Parameters Configuration
                    "architectures": self.pretrained_model,
                    "problem_type": "single_label_classification",
                    "finetuning_task": "asa_hpi_classification",
                    "asa_num_labels": self.asa_num_labels,
                    "emergency_num_labels": self.emergency_num_labels,
                    "asa_classifier_head_weight": self.asa_classifier_head_weight,
                    "emergency_classifier_head_weight": self.emergency_classifier_head_weight,
                    "label_smoothing_alpha": self.label_smoothing_alpha,
                    ### Tokenizer Parameters Configuration
                    "tokenizer_class": type(self.tokenizer).__name__,
                    "padding_side": self.tokenizer.padding_side,
                    "pad_token": self.tokenizer.vocab["[PAD]"],
                    "sep_token_id": self.tokenizer.vocab["[SEP]"],
                    "decoder_start_token_id": self.tokenizer.vocab["[CLS]"],
                    "unk_token_id": self.tokenizer.vocab["[UNK]"],
                    ### Tensor Data Type for PyTorch
                    "torch_dtype": self.torch_dtype,
                }
            )
            self.bert_config = bert_config

    def configure_model(self) -> None:
        self.model = BertForSequenceClassificationForASAPrediction.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model,
            config=self.bert_config,
            # torch_dtype=self.torch_dtype,
        )

    def configure_optimizers(self):
        "Define AdamW optimizer with optional 1-cycle learning rate schedule."
        if self.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                eps=self.adam_eps,
            )
        elif self.optimizer == "TransformersAdamW":
            self.optimizer = TransformersAdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                eps=self.adam_eps,
            )
        elif (self.optimizer is None) or (self.optimizer == ""):
            raise ValueError(
                "Must Specify `optimizer` arg by passing in a valid string "
                "value or a torch.optim.Optimizer."
            )

        # Note: when OneCycle schedule is used, `learning_rate` is the max learning rate
        if self.learning_rate_schedule == "OneCycle":
            train_dataset_length = len(self.trainer.datamodule.train)
            steps_per_epoch = 1 + int(
                train_dataset_length / (self.batch_size * self.accumulate_grad_batches)
            )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                anneal_strategy="cos",
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
            )
            lr_scheduler_config = {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
            return lr_scheduler_config
        else:
            self.scheduler = None
            return self.optimizer

    def shared_step(self, batch, batch_idx=None, class_weighted_loss=True):
        "Method for passing inputs into model to get outputs."
        outputs = self.model(
            input_ids=batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
            asa_labels=batch["asa_label"],
            emergency_labels=batch["emergency_label"],
            asa_class_weights=self.asa_class_weights.to(device=self.device)
            if class_weighted_loss
            else None,
            emergency_class_weights=self.emergency_class_weights.to(device=self.device)
            if class_weighted_loss
            else None,
        )
        # Add "loss" key in dict, which is required by lightning as output for training_step
        outputs = {**dict(outputs), "loss": outputs.combined_loss}
        # Compute Predictions from Logits
        outputs = {
            **outputs,
            "asa_predictions": torch.argmax(outputs["asa_logits"], dim=1),
            "emergency_predictions": torch.argmax(outputs["emergency_logits"], dim=1),
            "batch_idx": batch_idx,
        }
        return outputs

    def forward(self, batch, *args, **kwargs):
        "Method used for inference input -> output."
        return self.shared_step(batch)

    def training_step(self, batch, batch_idx):
        "Single training step.  Return dict must have `loss` key."
        outputs = self.shared_step(batch=batch, batch_idx=batch_idx)
        # "loss" must remain attached tensor so loss.backward() can be called by lightning
        return {k: detach(v) if k != "loss" else v for k, v in outputs.items()}

    def training_epoch_end(self, outputs: list[dict]):
        """End of train epoch.  Combine results from all train steps.

        Outputs is a List of dict of outputs from `train_step`.
        List(
            Dict(
                "batch_idx": int batch index,
                "loss": torch.tensor with loss from batch,
                ...
            ),
            ...
        )

        Must override to trigger `on_train_epoch_end` callback.
        """
        pass

    def validation_step(self, batch, batch_idx):
        "Single validation step.  Metrics logged during this step."
        outputs = self.shared_step(batch=batch, batch_idx=batch_idx)
        # "loss" must remain attached tensor so loss.backward() can be called by lightning
        return {k: detach(v) if k != "loss" else v for k, v in outputs.items()}

    def validation_epoch_end(self, outputs: list[dict]):
        """End of validation epoch.  Combine results from all validation steps.

        Outputs is a List of dict of outputs from `validation_step`.
        List(
            Dict(
                "batch_idx": int batch index,
                "loss": torch.tensor with loss from batch,
                ...
            ),
            ...
        )

        Must override to trigger `on_validation_epoch_end` callback.
        """
        pass

    def test_step(self, batch, batch_idx):
        "Single test step."
        outputs = self.shared_step(batch=batch, batch_idx=batch_idx)
        # "loss" must remain attached tensor so loss.backward() can be called by lightning
        return {k: detach(v) if k != "loss" else v for k, v in outputs.items()}

    def test_epoch_end(self, outputs: list[dict]):
        """End of test epoch.  Combine results from all test steps.

        Outputs is a List of dict of outputs from `test_step`.
        List(
            Dict(
                "batch_idx": int batch index,
                "loss": torch.tensor with loss from batch,
                ...
            ),
            ...
        )

        Must override to trigger `on_test_epoch_end` callback.
        """
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        outputs = self.shared_step(batch=batch, batch_idx=batch_idx)
        # "loss" must remain attached tensor so loss.backward() can be called by lightning
        return {k: detach(v) if k != "loss" else v for k, v in outputs.items()}
