from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import hydra
import mlflow
from azureml.core import Workspace
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from omegaconf import DictConfig, OmegaConf

from models.rf.test import test_rf_model_with_mlflow
from models.rf.train import train_rf_model_with_mlflow
from models.utils import resolve_paths_and_save_config
from src.modeling.mlflow import (
    get_best_child_run,
    log_mlflow_model,
    set_default_mlflow_tags,
)
from src.utils import in_notebook

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
        cfg = hydra.compose(config_name="rf_config") if cfg is None else cfg
        train_and_evaluate_rf_model_with_mlflow(cfg)
        return cfg


@hydra.main(
    config_path="../conf",
    config_name="rf_config",
    version_base=None,
)
def train_evaluate_hydra(cfg: DictConfig):
    "Train Model on command line using default config."
    train_and_evaluate_rf_model_with_mlflow(cfg)


def train_and_evaluate_rf_model_with_mlflow(
    cfg: Union[dict, DictConfig], use_best_child_run_params: bool = True
) -> dict[Any]:
    """Train Model (optionally find best child run from tuning experiment), Evaluate on Test Set.

    Args:
        cfg (DictConfig, dict): hydra configuration
        use_best_child_run_params (bool): if True, finds best child run from
            a hyperparameter tuning experiment or parent run.  Will search for
            child runs under `cfg.evaluate.experiment_name` if there is only one
            parent tuning run in the experiment.  If there are multiple parent tuning runs
            in the experiment, then `cfg.evaluate.parent_run_id` must be specified, and
            only child runs under that parent run will be searched.  The model parameters
            for the best child run will be used to train the model instead of the model
            parameters specified in hydra configuration.  If False, then model will be
            trained using parameters specified in hydra configuration.

    Returns:
        dict with keys `pipeline`, `model`, `datamodule`, `train_mlflow_run`, `train_tags`,
        `val_outputs`, `val_metrics`, `model_info`, `test_outputs`, `test_metrics`,
        `test_mlflow_run`, `test_tags`.
    """
    if use_best_child_run_params:
        # Get best child run
        best_child_run, best_child_run_params = get_best_child_run(cfg)
        best_child_run.info.run_id

        # Remove best child run params that are not in the original config and were added after job launch
        best_child_run_params = {
            k: v
            for k, v in best_child_run_params.items()
            if k in cfg.model.params.keys()
        }
        # Assign best child run params as the params to use for our next run
        cfg.model.params = best_child_run_params

    # Set New Experiment Name & Log Model to MLFlow
    cfg.mlflow.experiment_name = f"{cfg.evaluate.experiment_name}-eval"
    # cfg.mlflow.log_model = True

    # Full Config with manual overides from script
    log.debug(OmegaConf.to_yaml(cfg, resolve=True))

    # Start New Training & Evaluation
    ws = Workspace.from_config()
    mlflow_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    tags = {"ModelType": "RandomForest"}
    experiment_id = mlflow.create_experiment(
        cfg.mlflow.experiment_name,
        tags=tags,
    )
    with mlflow.start_run(
        experiment_id=experiment_id,
        description="Train & evaluate model with best hyperparameters from tuning.",
    ) as parent_run:
        # Log Tags, Config, Model Params, Code; Resolve Paths
        set_default_mlflow_tags(tags)
        cfg = resolve_paths_and_save_config(cfg, file_ref=__file__)
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True), artifact_file="config.yaml"
        )
        mlflow.log_params(cfg.model.params)
        mlflow.log_artifact(Path(__file__).parent.as_posix(), artifact_path="code")

        # Train Model
        with mlflow.start_run(
            experiment_id=experiment_id,
            description="Train model with best hyperparameters from tuning.",
            run_name="train_best",
            nested=True,
            tags={MLFLOW_PARENT_RUN_ID: parent_run.info.run_id},
        ) as train_run:
            results = train_rf_model_with_mlflow(cfg, log_model=False, create_run=False)
            datamodule = results["datamodule"]
            model = results["model"]
            val_outputs = results["outputs"]
            val_metrics = results["val_metrics"]
            pipeline = val_outputs["pipeline"]
            train_tags = results["tags"]
            model_info = results["model_info"]

        # Evaluate Model on Test Dataset
        with mlflow.start_run(
            experiment_id=experiment_id,
            description="Evaluate best model on test set.",
            run_name="test_best",
            nested=True,
            tags={MLFLOW_PARENT_RUN_ID: parent_run.info.run_id},
        ) as test_run:
            test_results = test_rf_model_with_mlflow(
                cfg=cfg, pipeline=pipeline, datamodule=datamodule, create_run=False
            )
            test_outputs = test_results["outputs"]
            test_metrics = test_results["metrics"]
            test_tags = test_results["tags"]

        # Model Logging
        with mlflow.start_run(
            experiment_id=experiment_id,
            description="Log trained model with best hyperparameters from tuning.",
            run_name="model_best",
            nested=True,
            tags={MLFLOW_PARENT_RUN_ID: parent_run.info.run_id},
        ) as model_log_run:
            model_log_run_tags = set_default_mlflow_tags(
                {**tags, "RunType": "ModelLogging"}
            )
            model_info = log_mlflow_model(
                model=pipeline,
                model_type="rf",
                project_root=cfg.general.project_root,
                code_paths=[Path(__file__).parent.as_posix()],
                registered_model_name=cfg.mlflow.registered_model_name,
                datamodule=datamodule,
                input_name=cfg.model.data.input_feature_name,
                output_name=cfg.model.data.output_label_name,
            )

    return {
        "pipeline": pipeline,
        "model": model,
        "datamodule": datamodule,
        "val_outputs": val_outputs,
        "val_metrics": val_metrics,
        "test_outputs": test_outputs,
        "test_metrics": test_metrics,
        "train_mlflow_run": train_run,
        "test_mlflow_run": test_run,
        "model_log_mlflow_run": model_log_run,
        "train_tags": train_tags,
        "test_tags": test_tags,
        "model_log_run_tags": model_log_run_tags,
        "model_info": model_info,
    }


if __name__ == "__main__":
    train_evaluate()
