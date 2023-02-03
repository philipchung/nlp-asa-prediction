from __future__ import annotations

import logging
from typing import Union

import hydra
from models.baseline.age_classifier import (
    train_age_classifier_model_and_evaluate_with_mlflow,
)
from models.baseline.random_classifier import (
    train_random_classifier_model_and_evaluate_with_mlflow,
)
from omegaconf import DictConfig, OmegaConf
from src.utils.notebook_utils import in_notebook

az_http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
az_http_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)


def train_evaluate(cfg: Union[dict, DictConfig] = None):
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    if in_notebook():
        # Running in notebook, hydra will not manage outputs
        train_evaluate_notebook(cfg)
    else:
        # Running on command line, hydra will manage output directory
        train_evaluate_hydra(cfg)


def train_evaluate_notebook(cfg: DictConfig = None):
    "Train Model in notebook.  Uses default config if arg `cfg` is not specified."
    with hydra.initialize(config_path="../conf", version_base=None):
        cfg = hydra.compose(config_name="baseline_config") if cfg is None else cfg
        train_random_classifier_model_and_evaluate_with_mlflow(cfg)
        train_age_classifier_model_and_evaluate_with_mlflow(cfg)
        return cfg


@hydra.main(config_path="../conf", config_name="baseline_config", version_base=None)
def train_evaluate_hydra(cfg: DictConfig):
    "Train Model on command line using default config."
    train_random_classifier_model_and_evaluate_with_mlflow(cfg)
    train_age_classifier_model_and_evaluate_with_mlflow(cfg)


if __name__ == "__main__":
    train_evaluate()
