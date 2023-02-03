from __future__ import annotations

from typing import Union

from omegaconf import DictConfig
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers.logger import Logger
from src.modeling.lightning_callbacks import Counter, FunctionalMetrics, LossMonitor


def make_lightning_trainer_with_callbacks(
    cfg: DictConfig,
    logger: Union[Logger, list[Logger]] = None,
    callbacks: Union[Callback, list[Callback]] = None,
    additional_callbacks: Union[Callback, list[Callback]] = [],
    eval_metric_split: str = "validation",
) -> Trainer:
    """
    Create a default pytorch lightning trainer with default configurations.
    """
    if callbacks is None:
        # Default callbacks
        callbacks = [
            RichProgressBar(),
            RichModelSummary(),
            # DeviceStatsMonitor(cpu_stats=False),
            ModelCheckpoint(
                dirpath=cfg.general.checkpoint_dir,
                **cfg.lightning.model_checkpoint,
            ),
            LearningRateMonitor(log_momentum=True),
            Counter(what="epochs"),
            Counter(what="batches"),
            Counter(what="num_examples"),
            LossMonitor(what="Combined", stage="all"),
            LossMonitor(what="ASA", stage="all"),
            LossMonitor(what="Emergency", stage="all"),
            FunctionalMetrics(
                what="ASA",
                split=eval_metric_split,
                k=cfg.metrics.examples_top_k,
                gpu=True,
                whitelist=cfg.metrics.whitelist,
            ),
            FunctionalMetrics(
                what="Emergency",
                split=eval_metric_split,
                k=cfg.metrics.examples_top_k,
                gpu=True,
                whitelist=cfg.metrics.whitelist,
            ),
            # StochasticWeightAveraging(
            #     swa_lrs=cfg.model.params.swa_lrs,
            #     swa_epoch_start=cfg.model.params.swa_epoch_start,
            # ),
        ]
    elif isinstance(callbacks, Callback):
        callbacks = [callbacks]

    if isinstance(additional_callbacks, Callback):
        additional_callbacks = [additional_callbacks]

    # Add additional callbacks
    callbacks = callbacks + additional_callbacks

    # Create Trainer
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        **cfg.lightning.trainer,
    )
    return trainer
