#%%
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Union

import hydra
import mlflow
import numpy as np
import pandas as pd
import shap
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
# ## Load Model, SHAP explanations from MLflow
#%%
# Specify model type & task
model_type = "bioclinicalbert"
task_name = "note512-asa"

# Configure MLflow
ws = Workspace.from_config()
mlflow_uri = ws.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(mlflow_uri)

# Get model runs
runs_dict = get_mlflow_runs(child_run_types=["model_runs", "shap_runs"])
model_runs_df = runs_dict["model_runs"]
shap_runs_df = runs_dict["shap_runs"]

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

#%%
# Get specific run where SHAP explanation file is logged as an artifact
shap_run = shap_runs_df.loc[model_type, task_input]

# Download SHAP explanation for model if does not exist locally
filename = f"{model_name}_v{model_version}_shap_explanation.pkl"
shap_artifact_path = Path("shap") / filename
local_dir_path = Path(__file__).parent
shap_local_path = Path(__file__).parent / shap_artifact_path
if not shap_local_path.exists():
    mlflow.artifacts.download_artifacts(
        run_id=shap_run.info.run_id,
        artifact_path=shap_artifact_path.as_posix(),
        dst_path=local_dir_path.as_posix(),
    )

# Load SHAP explanation
explanations = pickle.load(open(shap_local_path, "rb"))

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
# ## Error Analysis with SHAP explanations
# We now have asa labels, predictions, and shap explanations for all of the test set

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


def select_example_and_explanation(
    idx: int, examples: pd.DataFrame, shap_explanations: shap.Explanation
) -> tuple:
    example = examples.loc[idx, :]
    test_set_index = example.test_set_index.item()
    shap_explanation = shap_explanations[test_set_index, :, :]
    return example, shap_explanation


def shap_explanation_for_each_class(
    example: pd.Series,
    shap_explanation: shap.Explanation,
    class_names: list[str],
    save_html: bool,
):
    """Visualize SHAP explanations for each class for a single example.

    Args:
        example (pd.Series): single example (row in dataframe)
        shap_explanation (shap.Explanation): shap explanation object for example
        class_names (list[str]): list of class names
        save_html (bool): whether to save html
    """
    "Prints SHAP explanations.  If save_html=True, then will instead save to files."
    test_set_index = example.test_set_index.item()
    print(
        f"Index: {example.name}.  "
        f"Test Set Index: {test_set_index}.  "
        f"Original Dataset Index: {example.original_index}"
    )
    print(f"Full ASA: {example.ASA}")
    print(f"Anesthesiologist: {example.target}")
    print(f"Model Prediction: {example.prediction}")

    for class_index, class_label in enumerate(class_names):
        if save_html:
            html_figure = shap.plots.text(
                shap_explanation[:, class_index], display=False
            )
            filename = (
                f"figure_shap_{example.original_index}_"
                f"pred_{example.prediction}_label_{example.target}_"
                f"class_{class_label}.html"
            )
            with open(filename, "w") as file:
                file.write(html_figure)
        else:
            print(f"ASA {class_label}")
            shap.plots.text(shap_explanation[:, class_index])


#%% [markdown]
# ## Specific Examples, Manually De-identified
#
# Manual Deidentify based on HIPAA safe-harbor method.
# Age & date components shifted, names/locations/entities replaced with pseudonyms.
#%% [markdown]
# Predict ASA I, Target ASA IV-V
# Item Number: 0
# Test Set Index: 384.  Original Dataset Index: 57482
#%%
example, example_explanation = select_example_and_explanation(
    idx=0, examples=pred1_target4_df, shap_explanations=explanations
)

substitutions = {
    7: "6 ",
    9: "25 ",
    11: "2018 ",
    43: "16 ",
    56: "6 ",
    58: "25 ",
    60: "2018 ",
    62: "al",
    63: "li",
    64: "son ",
    72: "6 ",
    74: "24 ",
    104: "al",
    105: "li",
    106: "son",
    113: "al",
    114: "li",
    115: "son ",
    142: "al",
    143: "li",
    144: "son ",
    167: "mil",
    168: "wau",
    169: "kee ",
    280: "air",
    281: "lift ",
    282: "services ",
    284: "lake",
    285: "view ",
    294: "6 ",
    296: "25 ",
    316: "lake",
    317: "view ",
}
new_tokens = []
for idx, token in enumerate(example_explanation.data):
    if idx in substitutions:
        token = substitutions[idx]
    new_tokens += [token]

deidentified_example_explanation = copy.deepcopy(example_explanation)
deidentified_example_explanation.data = new_tokens
shap_explanation_for_each_class(
    example=example,
    shap_explanation=deidentified_example_explanation,
    class_names=datamodule.asa_class_names,
    save_html=True,
)

#%% [markdown]
# Predict ASA I, Target ASA IV-V
# Item Number: 2
# Test Set Index: 1285.  Original Dataset Index: 41739
#%%
example, example_explanation = select_example_and_explanation(
    idx=2, examples=pred1_target4_df, shap_explanations=explanations
)

substitutions = {
    7: "8 ",
    9: "17 ",
    11: "2019 ",
    47: "34 ",
    56: "34 ",
    212: "8 ",
    214: "17 ",
    216: "2019 ",
}
new_tokens = []
for idx, token in enumerate(example_explanation.data):
    if idx in substitutions:
        token = substitutions[idx]
    new_tokens += [token]

deidentified_example_explanation = copy.deepcopy(example_explanation)
deidentified_example_explanation.data = new_tokens
shap_explanation_for_each_class(
    example=example,
    shap_explanation=deidentified_example_explanation,
    class_names=datamodule.asa_class_names,
    save_html=True,
)

#%% [markdown]
# Predict ASA IV-V, Target ASA I
# Item Number: 12
# Test Set Index: 5577.  Original Dataset Index: 11950
#%%
example, example_explanation = select_example_and_explanation(
    idx=12, examples=pred4_target1_df, shap_explanations=explanations
)

substitutions = {
    7: "5 ",
    9: "10 ",
    11: "2020 ",
    54: "21 ",
}
new_tokens = []
for idx, token in enumerate(example_explanation.data):
    if idx in substitutions:
        token = substitutions[idx]
    new_tokens += [token]

deidentified_example_explanation = copy.deepcopy(example_explanation)
deidentified_example_explanation.data = new_tokens
shap_explanation_for_each_class(
    example=example,
    shap_explanation=deidentified_example_explanation,
    class_names=datamodule.asa_class_names,
    save_html=True,
)

#%% [markdown]
# Predict ASA IV-V, Target ASA I
# Item Number: 15
# Test Set Index: 7667.  Original Dataset Index: 29054
#%%
example, example_explanation = select_example_and_explanation(
    idx=15, examples=pred4_target1_df, shap_explanations=explanations
)

substitutions = {
    7: "9 ",
    9: "4 ",
    11: "2018 ",
    55: "j",
    56: "ef",
    57: "frey ",
    74: "3 ",
    76: "11 ",
    78: "2016 ",
    83: "46 ",
    96: "a",
    97: "ra",
    98: "hi",
    101: "6 ",
    103: "28 ",
    105: "2017 ",
    114: "j",
    115: "ef",
    116: "frey ",
    121: "46 ",
    251: "ch",
    252: "ana ",
    273: "2 ",
    275: "2016 ",
    282: "1 ",
    284: "2016 ",
    292: "5 ",
    294: "2016 ",
    298: "6 ",
    300: "2016 ",
    327: "2009 ",
    332: "1998 ",
    411: "9 ",
    413: "19 ",
    415: "2013 ",
    423: "3 ",
    425: "2016 ",
    455: "feb",
    456: "ru",
    457: "ary ",
    458: "2018",
    486: "2 ",
    488: "24 ",
    490: "2018 ",
}
new_tokens = []
for idx, token in enumerate(example_explanation.data):
    if idx in substitutions:
        token = substitutions[idx]
    new_tokens += [token]


deidentified_example_explanation = copy.deepcopy(example_explanation)
deidentified_example_explanation.data = new_tokens
shap_explanation_for_each_class(
    example=example,
    shap_explanation=deidentified_example_explanation,
    class_names=datamodule.asa_class_names,
    save_html=True,
)

# %%
