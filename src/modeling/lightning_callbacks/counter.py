from __future__ import annotations

from typing import Any

from pytorch_lightning.callbacks import Callback


class Counter(Callback):
    def __init__(self, what="epochs", verbose=True):
        self.what = what
        self.verbose = verbose
        self.state = {"epochs": 0, "batches": 0, "num_examples": 0}

    @property
    def state_key(self):
        # note: we do not include `verbose` here on purpose
        return self._generate_state_key(what=self.what)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.what == "epochs":
            self.state["epochs"] += 1
            self.log("Epochs", float(self.state["epochs"]))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.what == "batches":
            self.state["batches"] += 1
            self.log("Batches", float(self.state["batches"]))

        if self.what == "num_examples":
            self.state["num_examples"] += pl_module.batch_size
            self.log("NumExamples", float(self.state["num_examples"]))

    def state_dict(self) -> dict[str, Any]:
        return {
            "what": self.what,
            "verbose": self.verbose,
            "state": self.state,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for k, v in state_dict.items():
            setattr(self, k, v)
