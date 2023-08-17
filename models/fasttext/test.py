from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import fasttext
import mlflow
from azureml.core import Workspace
from models.utils import resolve_paths_and_save_config
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from src.metrics.format import format_all
from src.metrics.functional import all_metrics
from src.metrics.mlflow import log_metrics
from src.modeling.datamodule import DataModule
from src.modeling.fasttext import evaluate_fasttext
from src.modeling.mlflow import set_default_mlflow_tags

az_http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
az_http_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)


def test_fasttext_model(
    cfg: Union[dict, DictConfig],
    model: fasttext.FastText._FastText,
    datamodule: DataModule,
) -> dict[Any]:
    "Evaluate Model on Test Set."
    # Reproducibility
    seed_everything(seed=cfg.general.seed, workers=True)

    # Evaluate Model on Test Dataset
    test_outputs = evaluate_fasttext(
        model=model,
        datamodule=datamodule,
        split="test",
        input_feature_name=cfg.model.data.input_feature_name,
        output_label_name=cfg.model.data.output_label_name,
        num_output_classes=datamodule.asa_num_classes,
    )
    # Compute All Metrics
    test_metrics = all_metrics(
        preds=test_outputs["preds"],
        pred_proba=test_outputs["pred_proba"],
        target=test_outputs["target"],
        score_kind="probabilities",
        include_loss=True,
        num_classes=datamodule.asa_num_classes,
        class_labels=datamodule.asa_class_names,
        top_k_easiest_hardest=cfg.metrics.examples_top_k,
        dataset=datamodule.test,
        id2label=datamodule.asa_id2label,
        whitelist=cfg.metrics.whitelist,
        bootstrap_std_error=cfg.metrics.bootstrap_std_error
    )
    test_metrics = format_all(
        metrics=test_metrics,
        class_labels=datamodule.asa_class_names,
        split="Test",
        target_name="ASA",
    )
    return {
        "test_outputs": test_outputs,
        "test_metrics": test_metrics,
    }


def test_fasttext_model_with_mlflow(
    cfg: Union[dict, DictConfig],
    model: fasttext.FastText._FastText,
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
    tags = {"ModelType": "FastText"}
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
        tags = set_default_mlflow_tags({**tags, "RunType": "Test"})
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True),
            artifact_file="config.yaml",
        )
        mlflow.log_params(cfg.model.params)
        mlflow.log_artifact(Path(__file__).parent.as_posix(), artifact_path="code")

    # Evaluate Model
    results = test_fasttext_model(cfg=cfg, model=model, datamodule=datamodule)
    test_outputs = results["test_outputs"]
    test_metrics = results["test_metrics"]

    # Log metrics to MLflow
    log_metrics(test_metrics, step=0)

    # End run only if we created one
    if bool(create_run):
        mlflow.end_run()

    return {
        "model": model,
        "datamodule": datamodule,
        "outputs": test_outputs,
        "metrics": test_metrics,
        "mlflow_run": run,
        "tags": tags,
    }
