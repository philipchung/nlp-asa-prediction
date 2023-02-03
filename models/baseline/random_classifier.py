from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Union

import mlflow
import numpy as np
import pandas as pd
from azureml.core import Workspace
from models.utils import prepare_datamodule, resolve_paths_and_save_config
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, seed_everything
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from src.metrics.format import format_all
from src.metrics.functional import all_metrics
from src.metrics.mlflow import log_metrics
from src.modeling.mlflow import set_default_mlflow_tags

az_http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
az_http_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)


def train_evaluate_dummy_classifier(
    model: BaseEstimator = None,
    datamodule: LightningDataModule = None,
    input_feature_name: str = "input_text",
    output_label_name: str = "asa_label",
) -> dict[str, Any]:
    # Define Data Splits
    train_df = datamodule.train_dataframe()
    val_df = datamodule.val_dataframe()
    test_df = datamodule.test_dataframe()
    X_train = train_df[input_feature_name].to_numpy(dtype=np.unicode_)
    y_train = train_df[output_label_name].to_numpy(dtype=np.int64)
    X_val = val_df[input_feature_name].to_numpy(dtype=np.unicode_)
    y_val = val_df[output_label_name].to_numpy(dtype=np.int64)
    X_test = test_df[input_feature_name].to_numpy(dtype=np.unicode_)
    y_test = test_df[output_label_name].to_numpy(dtype=np.int64)

    # Default Model if None Specified
    if model is None:
        model = DummyClassifier(strategy="uniform", random_state=42)

    # Fit Model
    model.fit(X_train, y_train)

    # Make Predictions on Val Dataset
    y_val_pred = model.predict(X_val)
    y_val_pred_proba = model.predict_proba(X_val)

    # Make Predictions on Test Dataset
    y_test_pred = model.predict(X_test)
    y_test_pred_proba = model.predict_proba(X_test)

    return {
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "y_val_pred": y_val_pred,
        "y_val_pred_proba": y_val_pred_proba,
        "y_test_pred": y_test_pred,
        "y_test_pred_proba": y_test_pred_proba,
    }


def train_random_classifier_model_and_evaluate(cfg: Union[dict, DictConfig]) -> dict:
    "Train Random Classifier Model, Evaluate on Validation and Test Set."
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    cfg = resolve_paths_and_save_config(cfg)

    # Reproducibility
    seed_everything(seed=cfg.general.seed, workers=True)

    # Prepare Data
    datamodule = prepare_datamodule(cfg=cfg)

    # Create Model
    model = DummyClassifier(strategy="uniform", random_state=cfg.general.seed)
    # Train Model & Evaluate on Validation and Test Datasets
    outputs = train_evaluate_dummy_classifier(model=model, datamodule=datamodule)
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
    # Compute All Metrics
    test_metrics = all_metrics(
        preds=outputs["y_test_pred"],
        pred_proba=outputs["y_test_pred_proba"],
        target=outputs["y_test"],
        score_kind="probabilities",
        include_loss=True,
        num_classes=datamodule.asa_num_classes,
        class_labels=datamodule.asa_class_names,
        top_k_easiest_hardest=cfg.metrics.examples_top_k,
        dataset=datamodule.test,
        id2label=datamodule.asa_id2label,
        whitelist=cfg.metrics.whitelist,
    )
    test_metrics = format_all(
        metrics=test_metrics,
        class_labels=datamodule.asa_class_names,
        split="Test",
        target_name="ASA",
    )

    return {
        "datamodule": datamodule,
        "model": model,
        "outputs": outputs,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def train_random_classifier_model_and_evaluate_with_mlflow(
    cfg: Union[dict, DictConfig]
) -> dict[Any]:
    "Train Random Classifier Model, Evaluate on Validation and Test Set, Log to MLflow."
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    log.debug("Config:\n", OmegaConf.to_yaml(cfg, resolve=True))

    ws = Workspace.from_config()
    mlflow_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    tags = {"ModelType": "RandomClassifier"}
    experiment_id = mlflow.create_experiment(
        cfg.mlflow.experiment_name,
        tags=tags,
    )
    with mlflow.start_run(
        experiment_id=experiment_id,
        description="Train & evaluate model with best hyperparameters from tuning.",
    ) as parent_run:
        # Log Tags, Config, Code; Resolve Paths
        set_default_mlflow_tags({**tags, "RunType": "Train"})
        cfg = resolve_paths_and_save_config(cfg, file_ref=__file__)
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True), artifact_file="config.yaml"
        )
        mlflow.log_artifact(Path(__file__).parent.as_posix(), artifact_path="code")

        # Train Model, Evaluate on Validation and Test Set
        results = train_random_classifier_model_and_evaluate(cfg)
        val_metrics = results["val_metrics"]
        test_metrics = results["test_metrics"]
        model = results["model"]
        datamodule = results["datamodule"]

        # Log Validation Results
        with mlflow.start_run(
            experiment_id=experiment_id,
            description="Random Classifier Validation Set Metrics",
            run_name="random_classifier_validation",
            nested=True,
        ) as validation_run:
            # Log to MLflow
            set_default_mlflow_tags({**tags, "RunType": "Validation"})
            log_metrics(val_metrics, step=0)

        # Log Test Results
        with mlflow.start_run(
            experiment_id=experiment_id,
            description="Random Classifier Test Set Metrics",
            run_name="random_classifier_test",
            nested=True,
        ) as test_run:
            # Log to MLflow
            set_default_mlflow_tags({**tags, "RunType": "Test"})
            log_metrics(test_metrics, step=0)

        # Log Model
        with mlflow.start_run(
            experiment_id=experiment_id,
            description="Log random classifier baseline model.",
            run_name="random_classifier_model_logging",
            nested=True,
        ) as model_logging_run:
            set_default_mlflow_tags({**tags, "RunType": "ModelLogging"})
            # Code Paths
            project_root = datamodule.project_dir.resolve()
            pip_requirements_path = (project_root / "requirements.txt").as_posix()
            code_paths = [Path(__file__).parent.as_posix()]
            registered_model_name = f"random_classifier_{cfg.task.task_name}"
            # MLModel Signature
            input_name = cfg.model.data.input_feature_name
            output_name = cfg.model.data.output_label_name
            train_df = datamodule.train_dataframe(columns=[input_name, output_name])
            input_data_sample = train_df[input_name].iloc[:5].to_frame()
            output_data_sample = train_df[output_name].iloc[:5]
            signature = mlflow.models.infer_signature(
                input_data_sample, output_data_sample
            )
            # Input Example
            input_example = pd.DataFrame(
                [
                    "37 year old man with left femur fracture, scheduled for ORIF.",
                    "60 year old woman undergoing routine colonoscopy.",
                ]
            )
            # Log Model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                pip_requirements=pip_requirements_path,
                code_paths=code_paths,
                registered_model_name=registered_model_name,
                signature=signature,
                input_example=input_example,
            )

    return {**results, "model_info": model_info}
