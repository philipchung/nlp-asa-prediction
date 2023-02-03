from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import hydra
import mlflow
from azureml.core import Workspace
from models.utils import prepare_datamodule, resolve_paths_and_save_config
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from sklearn.ensemble import RandomForestClassifier
from src.metrics.format import format_all
from src.metrics.functional import all_metrics
from src.metrics.mlflow import log_metrics
from src.modeling.mlflow import log_mlflow_model, set_default_mlflow_tags
from src.modeling.sklearn import train_validate_sklearn
from src.utils import in_notebook

az_http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
az_http_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)


def train(cfg: Union[dict, DictConfig] = None):
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    if in_notebook():
        # Running in notebook, hydra will not manage outputs
        train_notebook(cfg)
    else:
        # Running on command line, hydra will manage output directory
        train_hydra(cfg)


def train_notebook(cfg: DictConfig = None):
    "Train Model in notebook.  Uses default config if arg `cfg` is not specified."
    with hydra.initialize(config_path="../conf", version_base=None):
        cfg = hydra.compose(config_name="rf_config") if cfg is None else cfg
        train_rf_model_with_mlflow(cfg)
        return cfg


@hydra.main(config_path="../conf", config_name="rf_config", version_base=None)
def train_hydra(cfg: DictConfig):
    "Train Model on command line using default config."
    train_rf_model_with_mlflow(cfg)


def train_rf_model(cfg: Union[dict, DictConfig]) -> dict:
    "Train Model, Evaluate on Validation Set."
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    cfg = resolve_paths_and_save_config(cfg)

    # Reproducibility
    seed_everything(seed=cfg.general.seed, workers=True)

    # Prepare Data
    datamodule = prepare_datamodule(cfg=cfg)

    # Create Model
    model = RandomForestClassifier(**cfg.model.params)
    # Train Model & Evaluate on Validation Dataset
    outputs = train_validate_sklearn(
        model=model,
        datamodule=datamodule,
        input_feature_name=cfg.model.data.input_feature_name,
        output_label_name=cfg.model.data.output_label_name,
        model_type="rf",
    )
    # Compute All Validation Metrics
    val_metrics = all_metrics(
        preds=outputs["y_val_pred"],
        pred_proba=outputs["y_val_pred_proba"],
        target=outputs["y_val"],
        score_kind="probabilities",
        include_loss=True,
        num_classes=datamodule.asa_num_classes,
        class_labels=datamodule.asa_class_names,
        top_k_easiest_hardest=cfg.metrics.examples_top_k,
        dataset=datamodule.validation,
        id2label=datamodule.asa_id2label,
        whitelist=cfg.metrics.whitelist,
    )
    val_metrics = format_all(
        metrics=val_metrics,
        class_labels=datamodule.asa_class_names,
        split="Validation",
        target_name="ASA",
    )

    return {
        "datamodule": datamodule,
        "model": model,
        "outputs": outputs,
        "val_metrics": val_metrics,
    }


def train_rf_model_with_mlflow(
    cfg: Union[dict, DictConfig],
    log_config: bool = True,
    log_model: bool = None,
    create_run: Union[bool, dict] = True,
) -> dict[Any]:
    "Train Model, Evaluate on Validation Set, Log to MLflow."
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    log.debug("Config:\n", OmegaConf.to_yaml(cfg, resolve=True))

    # Configure MLflow
    ws = Workspace.from_config()
    mlflow_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    tags = {"ModelType": "RandomForest"}
    tags = {**cfg.mlflow.tags, **tags} if cfg.mlflow.tags else tags
    if create_run:
        # Create New Run
        description = (
            cfg.mlflow.description if cfg.mlflow.description else "Training run"
        )
        experiment_id = mlflow.create_experiment(cfg.mlflow.experiment_name, tags=tags)
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=cfg.mlflow.run_name,
            nested=cfg.mlflow.nested,
            tags=tags,
            description=description,
        )
    else:
        run = mlflow.active_run()
        if run is None:
            raise ReferenceError("No active run exists.")

    # Resolve Paths
    cfg = resolve_paths_and_save_config(cfg, file_ref=__file__)

    # Log Tags, Config, Model Params, Code;
    if log_config:
        tags = set_default_mlflow_tags({**tags, "RunType": "Train"})
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True), artifact_file="config.yaml"
        )
        mlflow.log_params(cfg.model.params)
        mlflow.log_artifact(Path(__file__).parent.as_posix(), artifact_path="code")

    # Train Model, Evaluate on Validation Set
    results = train_rf_model(cfg)
    datamodule = results["datamodule"]
    model = results["model"]
    outputs = results["outputs"]
    pipeline = outputs["pipeline"]
    val_metrics = results["val_metrics"]

    # Log metrics to MLflow
    log_metrics(val_metrics, step=0)

    # Log Model to MLflow
    log_model = cfg.mlflow.log_model if log_model is None else log_model
    if log_model:
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
    else:
        model_info = None

    # End run only if we created one
    if bool(create_run):
        mlflow.end_run()

    return {**results, "model_info": model_info, "mlflow_run": run, "tags": tags}


if __name__ == "__main__":
    train()
