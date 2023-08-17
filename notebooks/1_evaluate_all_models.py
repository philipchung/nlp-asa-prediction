# %% [markdown]
# ## Evaluate All Models
# Load best models + baseline models.
# Evaluate them on all examples in the test set.
# Save results to pickle file.
# %%
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from azureml.core import Workspace
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from results_dataclasses import ModelPredictions
from tqdm.auto import tqdm

from models.baseline.age_meds_classifier import prepare_meds_data
from models.bert.trainer import make_lightning_trainer_with_callbacks
from src.dataset.data import Data
from src.modeling.datamodule import DataModule
from src.modeling.fasttext.evaluate import parse_fasttext_prediction
from src.modeling.mlflow.fasttext_flavor import load_model as load_fasttext_model
from src.modeling.sklearn import test_sklearn
from src.modeling.utils import combine_batches

log = logging.getLogger(__name__)

seed_everything(seed=42, workers=True)

current_dir = Path(__file__).parent
# %% [markdown]
# ## Load Model
# %%
# Configure MLflow
ws = Workspace.from_config()
mlflow_uri = ws.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(mlflow_uri)

# Get model runs
# runs_dict = get_mlflow_runs(child_run_types=["model_runs"])
# model_runs_df = runs_dict["model_runs"]

# Get reference to best model versions
project_root_dir = Path(__file__).parent.parent.resolve()
best_model_versions_path = (
    project_root_dir / "models" / "interpret" / "best_model_versions.yaml"
)
best_model_versions = OmegaConf.load(best_model_versions_path)
# %%
predictions = {}
# Evaluate all models on all tasks
for model_type, model_versions_dict in (pbar1 := tqdm(best_model_versions.items())):
    pbar1.set_description(f"Model Type: {model_type}")

    for task_input, model_version in (pbar2 := tqdm(model_versions_dict.items())):
        pbar2.set_description(f"Task Input: {task_input}")
        log.info("Evaluating Model: {model_type}, Task: {task_input}.")

        # Initialize Predictions Dict for Task Type
        predictions[task_input] = (
            predictions[task_input] if task_input in predictions else {}
        )

        # Define Task Name & Model Name
        task_name = f"{task_input}-asa"
        model_name = f"{model_type}_{task_name}"

        # Load Hydra Config for Model & Task
        if model_type in ("age_meds_classifier", "random_classifier"):
            config_name = "baseline_config"
        else:
            config_name = f"{model_type}_config"
        with hydra.initialize(config_path="../models/conf", version_base=None):
            cfg = hydra.compose(
                config_name=config_name,
                overrides=[
                    "general.export_dataset=false",
                    f"task={task_name}",
                ],
            )

        # Load Data
        if model_type == "bioclinicalbert":
            kwargs = {
                "dataloader_batch_size": cfg.model.params.batch_size,
                "seq_max_length": cfg.model.data.seq_max_length,
                **cfg.datamodule,
            }
        else:
            kwargs = {**cfg.datamodule}
        datamodule = DataModule(project_dir=project_root_dir, **kwargs)
        datamodule.prepare_data()
        test_df = datamodule.test_dataframe()
        # Set Predict Dataset to Test Set
        datamodule.predict = datamodule.test

        # Ground Truth Labels
        asa_class_names = datamodule.asa_class_names
        asa_label2id = datamodule.asa_label2id
        labels = datamodule.test["asa_label"].cpu().numpy()

        # Load Model & Evaluate
        match model_type:
            case "rf":
                rf_model = mlflow.sklearn.load_model(
                    model_uri=f"models:/{model_name}/{model_version}"
                )
                rf_outputs = test_sklearn(
                    pipeline=rf_model,
                    datamodule=datamodule,
                    input_feature_name=cfg.model.data.input_feature_name,
                    output_label_name=cfg.model.data.output_label_name,
                    model_type="rf",
                )
                rf_preds = rf_outputs["y_test_pred"]
                rf_pred_proba = rf_outputs["y_test_pred_proba"]
                predictions[task_input] |= {"rf_preds": rf_preds}
                predictions[task_input] |= {"rf_pred_proba": rf_pred_proba}
            case "svm":
                svm_model = mlflow.sklearn.load_model(
                    model_uri=f"models:/{model_name}/{model_version}"
                )
                svm_outputs = test_sklearn(
                    pipeline=svm_model,
                    datamodule=datamodule,
                    input_feature_name=cfg.model.data.input_feature_name,
                    output_label_name=cfg.model.data.output_label_name,
                    model_type="svm",
                )
                svm_preds = svm_outputs["y_test_pred"]
                svm_pred_proba = torch.nn.functional.softmax(
                    torch.from_numpy(svm_outputs["y_test_pred_score"]), dim=1
                ).numpy()
                predictions[task_input] |= {"svm_preds": svm_preds}
                predictions[task_input] |= {"svm_pred_proba": svm_pred_proba}
            case "fasttext":
                fasttext_model = load_fasttext_model(
                    model_uri=f"models:/{model_name}/{model_version}"
                )
                strings = test_df.input_text
                outputs = [
                    fasttext_model.predict(text=sentence, k=4) for sentence in strings
                ]
                parsed_outputs = [
                    parse_fasttext_prediction(output) for output in outputs
                ]
                fasttext_pred_proba = np.array(
                    [output["pred_proba"] for output in parsed_outputs]
                )
                fasttext_preds = np.array(fasttext_pred_proba).argmax(axis=1)
                predictions[task_input] |= {"fasttext_preds": fasttext_preds}
                predictions[task_input] |= {"fasttext_pred_proba": fasttext_pred_proba}
            case "bioclinicalbert":
                bioclinicalbert_model = mlflow.pytorch.load_model(
                    model_uri=f"models:/{model_name}/{model_version}"
                )
                # we inc. batch size because no backward pass/gradients in pytorch eval mode
                datamodule.dataloader_batch_size = 320
                datamodule.dataloader_num_workers = 8
                # Make Predictions
                trainer = make_lightning_trainer_with_callbacks(
                    cfg, eval_metric_split="test"
                )
                result = trainer.predict(bioclinicalbert_model, datamodule)
                result = combine_batches(result)
                bioclinicalbert_preds = result["asa_predictions"].cpu().numpy()
                bioclinicalbert_pred_proba = torch.nn.functional.softmax(
                    result["asa_logits"].detach().cpu().double(), dim=1
                ).numpy()
                predictions[task_input] |= {
                    "bioclinicalbert_preds": bioclinicalbert_preds
                }
                predictions[task_input] |= {
                    "bioclinicalbert_pred_proba": bioclinicalbert_pred_proba
                }
            case "age_meds_classifier":
                age_meds_model = mlflow.sklearn.load_model(
                    model_uri=f"models:/{model_name}/{model_version}"
                )
                # Get Additional Meds Data
                meds_data_path = datamodule.data_dir / "raw" / "meds_prior_to_start.tsv"
                meds = prepare_meds_data(
                    data_path=meds_data_path, prn_handling="Categorical"
                )
                meds_data = meds["data"]
                meds_count = meds["count"]
                meds_feature_rxnorm = meds["feature_rxnorm"]
                rxnorm2name = meds["rxnorm2name"]
                # Get Age from Cases Table, Combine with Test Dataset
                data = Data(
                    project_dir=datamodule.project_dir,
                    datastore_blob_path=datamodule.data_blob_prefix,
                    dataset_name=datamodule.dataset_name,
                    seed=datamodule.seed,
                )
                cases = data.get_cases()
                age_meds_test_df = (
                    test_df.reset_index()
                    .set_index("ProcedureID")
                    .join(cases.Age)
                    .join(meds_count)
                    .join(meds_data)
                    .fillna(False)
                    .reset_index()
                    .set_index("index")
                )
                # Define input & output data
                feature_columns = ["Age", "MedsCount"] + meds_feature_rxnorm
                output_label_name = "asa_label"
                age_meds_X_test = age_meds_test_df.loc[:, feature_columns].to_numpy(
                    dtype=np.int64
                )
                age_meds_preds = age_meds_model.predict(age_meds_X_test)
                age_meds_pred_proba = age_meds_model.predict_proba(age_meds_X_test)
                # Make Predictions
                predictions[task_input] |= {"age_meds_preds": age_meds_preds}
                predictions[task_input] |= {"age_meds_pred_proba": age_meds_pred_proba}
            case "random_classifier":
                random_classifier_model = mlflow.sklearn.load_model(
                    model_uri=f"models:/{model_name}/{model_version}"
                )
                input_feature_name = "input_text"
                random_X_test = test_df[input_feature_name].to_numpy(dtype=np.unicode_)
                random_classifier_preds = random_classifier_model.predict(random_X_test)
                random_classifier_pred_proba = random_classifier_model.predict_proba(
                    random_X_test
                )
                predictions[task_input] |= {
                    "random_classifier_preds": random_classifier_preds
                }
                predictions[task_input] |= {
                    "random_classifier_pred_proba": random_classifier_pred_proba
                }
            case _:
                log.warning(
                    f"Cannot perform Model Evaluation for model type: {model_type}."
                )


# Wrap Model Predictions for Task in a DataClass
model_predictions = {
    task: ModelPredictions(
        asa_class_names=asa_class_names,
        asa_label2id=asa_label2id,
        labels=labels,
        **task_predictions,
    )
    for task, task_predictions in predictions.items()
}

# %%
# Save Model Predictions to Disk
model_predictions_path = current_dir / "model_predictions.pkl"
with open(model_predictions_path, "wb") as file:
    pickle.dump(model_predictions, file, protocol=pickle.HIGHEST_PROTOCOL)
# %%
# # Load Model Predictions from Disk
# with open(model_predictions_path, "rb") as file:
#     model_predictions = pickle.load(file)
# %%
