from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import hydra
import mlflow
from azureml.core import Workspace
from models.bert.trainer import make_lightning_trainer_with_callbacks
from models.utils import prepare_datamodule, resolve_paths_and_save_config
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers import Logger, MLFlowLogger
from src.modeling.bert import ClinicalBertModel
from src.modeling.mlflow import log_mlflow_model, set_default_mlflow_tags
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
        cfg = (
            hydra.compose(config_name="bioclinicalbert_config") if cfg is None else cfg
        )
        train_bert_model_with_mlflow(cfg)
        return cfg


@hydra.main(
    config_path="../conf",
    config_name="bioclinicalbert_config",
    version_base=None,
)
def train_hydra(cfg: DictConfig):
    "Train Model on command line using default config."
    train_bert_model_with_mlflow(cfg)


def train_bert_model(
    cfg: Union[dict, DictConfig],
    logger: Logger,
    additional_callbacks: Union[Callback, list[Callback]] = [],
) -> dict:
    "Train Model, Evaluate on Validation Set."
    # Reproducibility
    seed_everything(seed=cfg.general.seed, workers=True)

    # Prepare Data
    datamodule = prepare_datamodule(cfg=cfg)

    # Define Model
    model = ClinicalBertModel(
        asa_label2id=datamodule.asa_label2id,
        emergency_label2id=datamodule.emergency_label2id,
        asa_class_weights=datamodule.asa_class_weights,
        emergency_class_weights=datamodule.emergency_class_weights,
        tokenizer=datamodule.tokenizer,
        precision=cfg.lightning.trainer.precision,
        **cfg.model.params,
    )

    # Create Trainer
    trainer = make_lightning_trainer_with_callbacks(
        cfg=cfg,
        logger=logger,
        additional_callbacks=additional_callbacks,
        eval_metric_split="validation",
    )

    # Run Trainer
    trainer.fit(model=model, datamodule=datamodule)
    return {
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }


def train_bert_model_with_mlflow(
    cfg: Union[dict, DictConfig],
    additional_callbacks: Union[Callback, list[Callback]] = [],
    log_config: bool = True,
    log_model: bool = None,
    create_run: Union[bool, dict] = True,
) -> dict[Any]:
    "Train Model, Evaluate on Validation Set, Log to MLFlow."
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    log.debug("Config:\n", OmegaConf.to_yaml(cfg, resolve=True))

    # Configure MLflow
    ws = Workspace.from_config()
    mlflow_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    tags = {"ModelType": "BERT"}
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

    # Log Tags, Config, Model Params, Code
    if log_config:
        tags = set_default_mlflow_tags({**tags, "RunType": "Train"})
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True), artifact_file="config.yaml"
        )
        mlflow.log_params(cfg.model.params)
        mlflow.log_artifact(Path(__file__).parent.as_posix(), artifact_path="code")

    # Train Model, Evaluate on Validation Set
    mlf_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        run_name=tags["RunName"],
        tracking_uri=mlflow_uri,
        run_id=run.info.run_id,
    )
    results = train_bert_model(
        cfg, logger=mlf_logger, additional_callbacks=additional_callbacks
    )
    datamodule = results["datamodule"]
    model = results["model"]
    trainer = results["trainer"]

    # Log Model to MLflow
    log_model = cfg.mlflow.log_model if log_model is None else log_model
    if log_model:
        # Remove unpicklable reference to trainer prior to logging model to MLflow
        model.trainer = None
        model._trainer = None

        model_info = log_mlflow_model(
            model=model,
            model_type="bert",
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
