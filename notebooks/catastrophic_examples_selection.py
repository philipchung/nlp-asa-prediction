#%%
import copy
from pathlib import Path
from typing import Union

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from azureml.core import Workspace
from models.bert.trainer import make_lightning_trainer_with_callbacks
from models.interpret.mlflow_runs import get_mlflow_runs
from omegaconf import OmegaConf
from src.metrics.format import format_confusion_matrix
from src.metrics.functional import confusion_matrix_metrics
from src.modeling.datamodule import DataModule
from src.modeling.utils import combine_batches

#%% [markdown]
# ## Load Model
#%%
# Specify model type & task
model_type = "bioclinicalbert"
task_name = "note512-asa"

# Configure MLflow
ws = Workspace.from_config()
mlflow_uri = ws.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(mlflow_uri)

# Get model runs
runs_dict = get_mlflow_runs(child_run_types=["model_runs"])
model_runs_df = runs_dict["model_runs"]
# shap_runs_df = runs_dict["shap_runs"]

# Get reference to best model versions (for saved SHAP explanation filename)
project_root_dir = Path(__file__).parent.parent.resolve()
best_model_versions_path = (
    project_root_dir / "models" / "interpret" / "best_model_versions.yaml"
)
best_model_versions = OmegaConf.load(best_model_versions_path)

# Load Trained Model from MLflow
task_input = task_name.split("-")[0]
model_name = f"{model_type}_{task_name}"
model_version = best_model_versions[model_type][task_input]
model = mlflow.pytorch.load_model(model_uri=f"models:/{model_name}/{model_version}")


#%% [markdown]
# ## Load Dataset and Evaluate Model on Test Set
#%%
# Load default hydra config for configuring datamodule
with hydra.initialize(config_path="../models/conf", version_base=None):
    cfg = hydra.compose(
        config_name=f"{model_type}_config",
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
# Set Predict Dataset to Test Set
# we inc. batch size because no backward pass/gradients in pytorch eval mode
datamodule.predict = datamodule.test
datamodule.dataloader_batch_size = 320
datamodule.dataloader_num_workers = 8

# Make Predictions
trainer = make_lightning_trainer_with_callbacks(cfg, eval_metric_split="test")
result = trainer.predict(model, datamodule)
result = combine_batches(result)
asa_predictions = result["asa_predictions"]
asa_labels = datamodule.test["asa_label"]

#%% [markdown]
# ## Error Analysis
# We now have asa labels and predictions for all of the test set

# %%
# Generate Confusion Matrix
conf_mat_metrics = confusion_matrix_metrics(preds=asa_predictions, target=asa_labels)
conf_mat_figures = format_confusion_matrix(
    conf_mat_metrics, class_labels=datamodule.asa_class_names
)

#%%
def select_examples(
    selected_pred: str,
    selected_target: str,
    datamodule: DataModule,
    predictions: Union[torch.Tensor, np.array],
    targets: Union[torch.Tensor, np.array],
):
    """Get examples corresponding to a single cell in the confusion matrix.

    Args:
        selected_pred (str): predicted ASA class to select (model prediction)
        selected_target (str): actual ASA class to select (ground truth)
        datamodule (DataModule): datamodule object
        predictions (Union[torch.Tensor, np.array]): actual model predictions
        targets (Union[torch.Tensor, np.array]): actual ground truth labels

    Returns:
        _type_: _description_
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    selected = []
    index = datamodule.test["index"]
    pred_label = datamodule.asa_label2id[selected_pred]
    target_label = datamodule.asa_label2id[selected_target]
    for i, pred, target in zip(index, predictions, targets):
        if pred == pred_label and target == target_label:
            selected += [
                {
                    "index": i,
                    "prediction": datamodule.asa_id2label[pred],
                    "target": datamodule.asa_id2label[target],
                }
            ]
    selected_df = pd.DataFrame(selected)

    test_df = datamodule.test_dataframe().reset_index(
        drop=False, names="original_index"
    )
    # Note: test_df has a column "original_index" which is the sample number from
    # original dataset (prior to dataset split & shuffling).
    test_df = test_df.reset_index(drop=False, names="test_set_index")
    # "test_set_index" is the order in which examples appear in the test set,
    # which corresponds to the index in the SHAP explanation object.

    df = pd.merge(
        selected_df, test_df, left_on="index", right_on="original_index"
    ).drop(columns="index")
    return df


#%% [markdown]
# ## Get Only Catastrophic Error Examples
#%%
# Find all examples where Predicted=I, Actual=IV-V
pred1_target4_df = select_examples(
    selected_pred="I",
    selected_target="IV-V",
    datamodule=datamodule,
    predictions=asa_predictions,
    targets=asa_labels,
)

# Find all examples where Predicted=IV-V, Actual=I
pred4_target1_df = select_examples(
    selected_pred="IV-V",
    selected_target="I",
    datamodule=datamodule,
    predictions=asa_predictions,
    targets=asa_labels,
)
#%%
# Shuffle these Examples Together, Retain Index for Reference
catastrophic_errors_df = pd.concat([pred1_target4_df, pred4_target1_df], axis=0)
#%%
export_catastrophic_errors_df = catastrophic_errors_df.set_index("original_index").loc[
    :, ["prediction", "target", "input_text"]
]
#%%
# Shuffle examples
shuffled_catastrophic_errors_df = export_catastrophic_errors_df.sample(
    frac=1, random_state=42
)
# Copy to make answer key
shuffled_catastrophic_errors_answers_df = copy.deepcopy(shuffled_catastrophic_errors_df)
# Remove answers for survey form
shuffled_catastrophic_errors_df["Modified ASA-PS"] = ""
shuffled_catastrophic_errors_df = shuffled_catastrophic_errors_df.drop(
    columns=["prediction", "target"]
)
#%%
# Export Survey & Answer Key to Disk
shuffled_catastrophic_errors_answers_df.to_csv(
    "catastrophic_errors_answers.tsv", sep="\t"
)
shuffled_catastrophic_errors_df.to_csv("catastrophic_errors_survey.tsv", sep="\t")

#%% [markdown]
