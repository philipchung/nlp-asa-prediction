from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import mlflow
from azureml.core import Workspace
from models.bert.trainer import make_lightning_trainer_with_callbacks
from models.utils import resolve_paths_and_save_config
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import MLFlowLogger
from src.modeling.bert import ClinicalBertModel
from src.modeling.datamodule import DataModule
from src.modeling.mlflow import set_default_mlflow_tags

az_http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
az_http_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)


def test_bert_model(
    cfg: Union[dict, DictConfig],
    model: ClinicalBertModel,
    datamodule: DataModule,
    run_id: str,
) -> dict[Any]:
    "Evaluate Model on Test Set."
    # Reproducibility
    seed_everything(seed=cfg.general.seed, workers=True)

    # Create Trainer
    mlf_logger = MLFlowLogger(run_id=run_id)
    trainer = make_lightning_trainer_with_callbacks(
        cfg=cfg, logger=mlf_logger, eval_metric_split="test"
    )

    # Run Trainer
    trainer.test(model=model, datamodule=datamodule)
    return {"trainer": trainer}


def test_bert_model_with_mlflow(
    cfg: Union[dict, DictConfig],
    model: ClinicalBertModel,
    datamodule: DataModule,
    log_config: bool = True,
    create_run: bool = True,
) -> dict[Any]:
    "Evaluate Model on Test Set."
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg

    # Configure MLflow
    ws = Workspace.from_config()
    mlflow_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    tags = {"ModelType": "BERT"}
    tags = {**cfg.mlflow.tags, **tags} if cfg.mlflow.tags else tags
    # Create a New Run
    if bool(create_run):
        if isinstance(create_run, dict):
            # Create run with specific settings
            run = mlflow.start_run(**create_run)
        else:
            # Default run settings
            description = (
                cfg.mlflow.description if cfg.mlflow.description else "Test run"
            )
            experiment_id = mlflow.create_experiment(
                cfg.mlflow.experiment_name, tags=tags
            )
            run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=cfg.mlflow.run_name,
                nested=cfg.mlflow.nested,
                tags=tags,
                description=description,
            )
    # Or use existing active run
    else:
        run = mlflow.active_run()
        if run is None:
            raise ReferenceError("No active run exists.")

    # Resolve Paths
    cfg = resolve_paths_and_save_config(cfg, file_ref=__file__)

    # Log Tags, Config, Model Params, Code
    if log_config:
        set_default_mlflow_tags({**tags, "RunType": "Test"})
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True),
            artifact_file="config.yaml",
        )
        mlflow.log_params(cfg.model.params)
        mlflow.log_artifact(Path(__file__).parent.as_posix(), artifact_path="code")

    # Evaluate Model
    results = test_bert_model(
        cfg=cfg, model=model, datamodule=datamodule, run_id=run.info.run_id
    )
    trainer = results["trainer"]

    # End run only if we created one
    if bool(create_run):
        mlflow.end_run()

    return {
        "model": model,
        "datamodule": datamodule,
        "trainer": trainer,
        "mlflow_run": run,
        "tags": tags,
    }
