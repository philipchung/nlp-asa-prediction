from __future__ import annotations

from typing import Any, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from src.metrics.format.format import format_all
from src.metrics.functional.metrics import all_metrics
from src.metrics.mlflow import itemize_tensors, log_only_dataframes_and_figures


class FunctionalMetrics(Callback):
    """
    Pytorch-Lightning Callback that stores evaluation predictions and
    computes functional torchmetrics over them.

    Args:
        what (str): output variable for which metric is computed ("ASA", "Emergency").  Case sensitive.
        split (str or tuple or list): split(s) to log ("train", "validation", "test").
            "all" can be passed as shorthand for all splits.
        gpu (bool): whether to compute metric on GPU or CPU
        k (int): number of top easiest & top hardest examples per epoch to record in table
        whitelist (list, dict, or DictConfig): list of metric names to log.  If `None`, all metrics are logged.
    """

    def __init__(
        self,
        what: str = None,
        split: Union[str, tuple[str], list[str]] = "all",
        gpu: bool = True,
        k: int = 10,
        whitelist: Union[list[str], dict[str, Any], DictConfig[str, Any]] = None,
        bootstrap_std_error: bool = False,
    ):
        self.what = what
        self._target = what.lower()
        split = (split,) if isinstance(split, str) else split
        for s in split:
            if s not in ("train", "validation", "test", "all"):
                raise ValueError(
                    f"Unknown split {s} passed to FunctionalMetrics Callback initialization."
                )
        self.split = ("train", "validation", "test") if split == ("all",) else split
        self.gpu = gpu
        self.k = k
        self.whitelist = whitelist
        self.bootstrap_std_error = bootstrap_std_error
        self.state = {
            "train": {"labels": [], "logits": [], "predictions": []},
            "validation": {"labels": [], "logits": [], "predictions": []},
            "test": {"labels": [], "logits": [], "predictions": []},
            "predict": {"labels": [], "logits": [], "predictions": []},
        }

    @property
    def state_key(self):
        return self._generate_state_key(what=self.what)

    def setup(self, trainer, pl_module, stage):
        # Derive properties from datamodule
        self.num_classes = getattr(trainer.datamodule, f"{self._target}_num_classes")
        self.class_names = getattr(trainer.datamodule, f"{self._target}_class_names")
        self.label2id = dict(getattr(trainer.datamodule, f"{self._target}_label2id"))
        self.id2label = {v: k for k, v in self.label2id.items()}

    def _on_batch_end(self, split, outputs, batch):
        # Get output from each batch
        labels = batch[f"{self._target}_label"]
        logits = outputs[f"{self._target}_logits"]
        predictions = outputs[f"{self._target}_predictions"]
        # Accumulate labels, logits, predictions
        self.state[split]["labels"] += [labels]
        self.state[split]["logits"] += [logits]
        self.state[split]["predictions"] += [predictions]

    def _on_epoch_end(self, split, trainer, pl_module):
        # Combine from all batches
        labels = torch.cat(self.state[split]["labels"])
        logits = torch.cat(self.state[split]["logits"])
        predictions = torch.cat(self.state[split]["predictions"])
        if self.gpu:
            labels = labels.cuda()
            logits = logits.cuda()
            predictions = predictions.cuda()
        else:
            labels = labels.cpu()
            logits = logits.cpu()
            predictions = predictions.cpu()

        # Compute all functional metrics
        metrics = all_metrics(
            preds=predictions,
            pred_proba=logits,
            target=labels,
            score_kind="logits",
            include_loss=False,
            num_classes=self.num_classes,
            class_labels=self.class_names,
            top_k_easiest_hardest=self.k,
            dataset=getattr(trainer.datamodule, split),
            id2label=self.id2label,
            whitelist=self.whitelist,
            bootstrap_std_error=self.bootstrap_std_error,
        )
        metrics = format_all(
            metrics=metrics,
            class_labels=self.class_names,
            split=split.capitalize(),
            target_name=self.what,
        )
        # Log dataframes and figures directly with MLflow because not supported by pytorch-lightning MLflow logger
        metrics = log_only_dataframes_and_figures(metrics, step=trainer.global_step)

        # Log with pytorch-lightning MLflow logger
        metrics = itemize_tensors(metrics, make_float=True)
        self.log_dict(metrics)

        # Reset accumulated labels, logits, predictions after each epoch
        self.state[split]["labels"] = []
        self.state[split]["logits"] = []
        self.state[split]["predictions"] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "train" in self.split:
            self._on_batch_end("train", outputs, batch)

    def on_train_epoch_end(self, trainer, pl_module):
        if "train" in self.split:
            self._on_epoch_end("train", trainer, pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if "validation" in self.split:
            self._on_batch_end("validation", outputs, batch)

    def on_validation_epoch_end(self, trainer, pl_module):
        if "validation" in self.split:
            self._on_epoch_end("validation", trainer, pl_module)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if "test" in self.split:
            self._on_batch_end("test", outputs, batch)

    def on_test_epoch_end(self, trainer, pl_module):
        if "test" in self.split:
            self._on_epoch_end("test", trainer, pl_module)

    def state_dict(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "split": self.split,
            "gpu": self.gpu,
            "k": self.k,
            "whitelist": self.whitelist,
            "state": self.state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)
