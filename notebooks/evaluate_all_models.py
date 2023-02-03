#%% [markdown]
# ## Evaluate All Models
# Load best models + baseline models. Evaluate them on all examples in the test set. Save results to pickle file.
#%%
from __future__ import annotations

import pickle
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from azureml.core import Workspace
from models.bert.trainer import make_lightning_trainer_with_callbacks
from models.interpret.mlflow_runs import get_mlflow_runs
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from src.dataset.data import Data
from src.modeling.datamodule import DataModule
from src.modeling.fasttext.evaluate import parse_fasttext_prediction
from src.modeling.mlflow.fasttext_flavor import load_model as load_fasttext_model
from src.modeling.sklearn import test_sklearn
from src.modeling.utils import combine_batches

seed_everything(seed=42, workers=True)

#%% [markdown]
# ## Load Model
#%%
# Configure MLflow
ws = Workspace.from_config()
mlflow_uri = ws.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(mlflow_uri)

# Get model runs
runs_dict = get_mlflow_runs(child_run_types=["model_runs"])
model_runs_df = runs_dict["model_runs"]

# Get reference to best model versions
project_root_dir = Path(__file__).parent.parent.resolve()
best_model_versions_path = (
    project_root_dir / "models" / "interpret" / "best_model_versions.yaml"
)
best_model_versions = OmegaConf.load(best_model_versions_path)
#%%
# Load Trained Models from MLflow
model_type = "rf"
task_name = "note512-asa"
task_input = task_name.split("-")[0]
model_name = f"{model_type}_{task_name}"
model_version = best_model_versions[model_type][task_input]
rf_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

model_type = "svm"
task_name = "note512-asa"
task_input = task_name.split("-")[0]
model_name = f"{model_type}_{task_name}"
model_version = best_model_versions[model_type][task_input]
svm_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")

model_type = "fasttext"
task_name = "note512-asa"
task_input = task_name.split("-")[0]
model_name = f"{model_type}_{task_name}"
model_version = best_model_versions[model_type][task_input]
fasttext_model = load_fasttext_model(model_uri=f"models:/{model_name}/{model_version}")

model_type = "bioclinicalbert"
task_name = "note512-asa"
task_input = task_name.split("-")[0]
model_name = f"{model_type}_{task_name}"
model_version = best_model_versions[model_type][task_input]
bioclinicalbert_model = mlflow.pytorch.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

model_type = "age_classifier"
task_name = "note512-asa"
task_input = task_name.split("-")[0]
model_name = f"{model_type}_{task_name}"
model_version = best_model_versions[model_type][task_input]
age_classifier_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)

model_type = "random_classifier"
task_name = "note512-asa"
task_input = task_name.split("-")[0]
model_name = f"{model_type}_{task_name}"
model_version = best_model_versions[model_type][task_input]
random_classifier_model = mlflow.sklearn.load_model(
    model_uri=f"models:/{model_name}/{model_version}"
)
#%% [markdown]
# ## Load Dataset and Evaluate Model on Test Set
# After this is run once, you can just load the results from the saved pickle file
#%%
# Load default hydra config for configuring datamodule
with hydra.initialize(config_path="../models/conf", version_base=None):
    cfg = hydra.compose(
        config_name=f"bioclinicalbert_config",
        overrides=[
            "general.export_dataset=false",
            f"task={task_name}",
        ],
    )

# Load Data
datamodule = DataModule(
    project_dir=project_root_dir,
    dataloader_batch_size=cfg.model.params.batch_size,
    seq_max_length=cfg.model.data.seq_max_length,
    **cfg.datamodule,
)
datamodule.prepare_data()
test_df = datamodule.test_dataframe()
# Set Predict Dataset to Test Set
datamodule.predict = datamodule.test
#%%
## Model Evaluation
# Ground Truth Labels
results = {
    "asa_class_names": datamodule.asa_class_names,
    "asa_label2id": datamodule.asa_label2id,
    "labels": datamodule.test["asa_label"].cpu().numpy(),
}

## Evaluate RF
rf_outputs = test_sklearn(
    pipeline=rf_model,
    datamodule=datamodule,
    input_feature_name=cfg.model.data.input_feature_name,
    output_label_name=cfg.model.data.output_label_name,
    model_type="rf",
)
results["rf_preds"] = rf_outputs["y_test_pred"]
results["rf_pred_proba"] = rf_outputs["y_test_pred_proba"]


## Evaluate SVMf
svm_outputs = test_sklearn(
    pipeline=svm_model,
    datamodule=datamodule,
    input_feature_name=cfg.model.data.input_feature_name,
    output_label_name=cfg.model.data.output_label_name,
    model_type="svm",
)
results["svm_preds"] = svm_outputs["y_test_pred"]
results["svm_pred_proba"] = torch.nn.functional.softmax(
    torch.from_numpy(svm_outputs["y_test_pred_score"]), dim=1
).numpy()

## Evaluate fastText
strings = test_df.input_text
outputs = [fasttext_model.predict(text=sentence, k=4) for sentence in strings]
parsed_outputs = [parse_fasttext_prediction(output) for output in outputs]
fasttext_pred_proba = [output["pred_proba"] for output in parsed_outputs]
fasttext_preds = np.array(fasttext_pred_proba).argmax(axis=1)
results["fasttext_preds"] = fasttext_preds
results["fasttext_pred_proba"] = np.array(fasttext_pred_proba)

## Evaluate BioClinicalBERT
# we inc. batch size because no backward pass/gradients in pytorch eval mode
datamodule.dataloader_batch_size = 320
datamodule.dataloader_num_workers = 8
# Make Predictions
trainer = make_lightning_trainer_with_callbacks(cfg, eval_metric_split="test")
result = trainer.predict(bioclinicalbert_model, datamodule)
result = combine_batches(result)
results["bioclinicalbert_preds"] = result["asa_predictions"].cpu().numpy()
results["bioclinicalbert_pred_proba"] = torch.nn.functional.softmax(
    result["asa_logits"].detach().cpu().double(), dim=1
).numpy()

## Evaluate Age Classifier Baseline Model
# Get Age from Cases Table, Combine with Test Dataset
data = Data(
    project_dir=datamodule.project_dir,
    datastore_blob_path=datamodule.data_blob_prefix,
    dataset_name=datamodule.dataset_name,
    seed=datamodule.seed,
)
cases = data.get_cases()
age_test_df = (
    test_df.reset_index()
    .set_index("ProcedureID")
    .join(cases.Age)
    .reset_index()
    .set_index("index")
)
# Define input & output data
input_feature_name = "Age"
output_label_name = "asa_label"
age_X_test = age_test_df[input_feature_name].to_numpy(dtype=np.int64).reshape(-1, 1)
# Make Predictions on Test Dataset
age_y_test_pred = age_classifier_model.predict(age_X_test)
age_y_test_pred_proba = age_classifier_model.predict_proba(age_X_test)
results["age_classifier_preds"] = age_y_test_pred
results["age_classifier_pred_proba"] = age_y_test_pred_proba

## Evaluate Random Classifier Baseline Model
input_feature_name = "input_text"
random_X_test = test_df[input_feature_name].to_numpy(dtype=np.unicode_)
random_y_test_pred = random_classifier_model.predict(random_X_test)
random_y_test_pred_proba = random_classifier_model.predict_proba(random_X_test)
results["random_classifier_preds"] = random_y_test_pred
results["random_classifier_pred_proba"] = random_y_test_pred_proba

# Save Evaluation Results to Disk
with open("model_predictions.pkl", "wb") as file:
    pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
#%%
