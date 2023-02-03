from __future__ import annotations
from tkinter import FALSE

from typing import Any, Tuple, Union

import torch
from pytorch_lightning.callbacks import Callback


class LossMonitor(Callback):
    """
    Callback for logging loss.

    Args:
        what (str): which loss to compute metrics ("combined", "asa", "emergency").  Case sensitive.
        stage (str or tuple): stage(s) to log ("train", "validation", "test").
            "all" can be passed as shorthand for all stages.
        gpu (bool): whether to compute metric on GPU or CPU
        batch_loss (bool): whether to log loss for each batch in addition average across
            entire epoch
    """

    def __init__(
        self,
        what: str = None,
        stage: Union[str, Tuple[str]] = "all",
        gpu: bool = True,
        batch_loss: bool = FALSE,
    ):
        self.what = what if what else "combined"
        self.loss_name = f"{self.what.lower()}_loss"
        stage = (stage,) if not isinstance(stage, tuple) else stage
        for s in stage:
            if s not in ("train", "validation", "test", "all"):
                raise ValueError(
                    f"Unknown stage {s} passed to LossMonitor Callback initialization."
                )
        self.stage = ("train", "validation", "test") if stage == ("all",) else stage
        self.gpu = gpu
        self.batch_loss = batch_loss
        self.state = {
            "train": [],
            "validation": [],
            "test": [],
        }

    @property
    def state_key(self):
        return self._generate_state_key(stage=self.stage)

    def _on_batch_end(self, stage, trainer, outputs):
        loss = outputs[self.loss_name] if self.gpu else outputs[self.loss_name].cpu()
        self.state[stage] += [loss]
        if self.batch_loss:
            self.log(f"{self.what}/{stage.capitalize()}/BatchCrossEntropyLoss", loss)

    def _on_epoch_end(self, stage, trainer):
        epoch_loss = torch.tensor(self.state[stage]).mean()
        self.log(f"{self.what}/{stage.capitalize()}/CrossEntropyLoss", epoch_loss)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if "train" in self.stage:
            self._on_batch_end("train", trainer, outputs)

    def on_train_epoch_start(self, trainer, pl_module):
        if "train" in self.stage:
            self.state["train"] = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train" in self.stage:
            self._on_epoch_end("train", trainer)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if "validation" in self.stage:
            self._on_batch_end("validation", trainer, outputs)

    def on_validation_epoch_start(self, trainer, pl_module):
        if "validation" in self.stage:
            self.state["validation"] = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if "validation" in self.stage:
            self._on_epoch_end("validation", trainer)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if "test" in self.stage:
            self._on_batch_end("test", trainer, outputs)

    def on_test_epoch_start(self, trainer, pl_module):
        if "test" in self.stage:
            self.state["test"] = []

    def on_test_epoch_end(self, trainer, pl_module):
        if "test" in self.stage:
            self._on_epoch_end("test", trainer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "loss_name": self.loss_name,
            "stage": self.stage,
            "gpu": self.gpu,
            "batch_loss": self.batch_loss,
            "state": self.state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)
