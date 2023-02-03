from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Union

import hydra
import mlflow
import shap
from azureml.core import Workspace
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from models.interpret.mlflow_runs import get_mlflow_runs
from models.interpret.shap_clinicalbert_model import ShapClinicalBERTModelWrapper
from models.utils import resolve_paths_and_save_config
from omegaconf import DictConfig, OmegaConf
from src.modeling.datamodule import DataModule
from src.modeling.mlflow import set_default_mlflow_tags

log = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf",
    config_name="bioclinicalbert_shap_config.yaml",
    version_base=None,
)
def explain_bert_model(cfg: Union[dict, DictConfig]):
    "Train SHAP Explainer for fasttext model."
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    cfg = resolve_paths_and_save_config(cfg)

    ws = Workspace.from_config()
    mlflow_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)

    # Get Best Model Versions
    best_model_versions_path = (
        cfg.best_model_versions_path
        if cfg.best_model_versions_path
        else Path(cfg.general.project_root)
        / "models"
        / "interpret"
        / "best_model_versions.yaml"
    )
    best_model_versions = OmegaConf.load(best_model_versions_path)

    # Load Model from Model Registry
    model_type = cfg.model_type
    task_name = cfg.task_name
    task_input = task_name.split("-")[0]

    model_name = f"{model_type}_{task_name}"
    model_version = best_model_versions[model_type][task_input]
    model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")

    # Load Data
    datamodule = DataModule(
        project_dir=cfg.general.project_root,
        task_name=cfg.task_name,
        seq_max_length=cfg.seq_max_length,
        tokenizer=cfg.tokenizer,
        seed=cfg.seed,
    )
    datamodule.prepare_data()
    # Get text list from test set
    text_list = datamodule.test["input_text"]

    # Wrap Model for SHAP library
    shap_bert_model = ShapClinicalBERTModelWrapper(
        pl_model=model.to(device="cuda"), seq_max_length=512, output_score_type="asa"
    )

    # # Get model output scores (used internally by shap.Explainer)
    # scores = shap_bert_model(text_list)

    # Define masker
    masker = shap_bert_model.tokenizer

    # Create explainer using transformers pipeline (native API)
    explainer = shap.Explainer(
        model=shap_bert_model,
        masker=masker,
        link=shap.links.identity,
        algorithm="auto",
        seed=cfg.seed,
        output_names=datamodule.asa_class_names,
    )

    # Compute SHAP values.  This takes a long time
    explanation = explainer(text_list)

    # Save SHAP Explanation locally as pickled file
    filename = f"{model_name}_v{model_version}_shap_explanation.pkl"
    filepath = (
        Path(cfg.save_path) / filename
        if cfg.save_path
        else Path(cfg.general.output_dir) / filename
    )
    pickle.dump(explanation, open(filepath, "wb"))

    # Log SHAP Explanation to MLflow
    if cfg.log_explanation:
        # Get Evaluation MLflow runs
        runs_dict = get_mlflow_runs()
        parent_runs_df = runs_dict["parent_runs"]
        # Get Parent run for model type and task
        parent_run = parent_runs_df.loc[model_type, task_input]
        # Create new child run to log SHAP values
        explanation_run_name = (
            cfg.explanation_run_name if cfg.explanation_run_name is not None else "shap"
        )
        tags = {
            "ModelType": "BioClinicalBERT",
            "RunType": "Shap",
            MLFLOW_PARENT_RUN_ID: parent_run.info.run_id,
        }
        mlflow.start_run(
            experiment_id=parent_run.info.experiment_id,
            run_name=explanation_run_name,
            nested=True,
            tags=tags,
            description="Shap Explanations for best model.",
        )
        tags = set_default_mlflow_tags(tags)
        mlflow.log_artifact(local_path=filepath, artifact_path="shap")
        mlflow.end_run()


if __name__ == "__main__":
    explain_bert_model()
