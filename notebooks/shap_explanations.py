#%%
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import shap
from azureml.core import Workspace
from models.interpret.mlflow_runs import get_mlflow_runs
from omegaconf import OmegaConf
from src.modeling.datamodule import DataModule

#%% [markdown]
# ## Load SHAP explanations from MLflow
#%%
# Specify model type & task
model_type = "bioclinicalbert"
task_name = "note512-asa"

# Configure MLflow
ws = Workspace.from_config()
mlflow_uri = ws.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(mlflow_uri)

# Get model runs
runs_dict = get_mlflow_runs(child_run_types="model_runs")
model_runs_df = runs_dict["model_runs"]

# Get reference to best model versions (for saved SHAP explanation filename)
project_root_dir = Path(__file__).parent.parent.resolve()
best_model_versions_path = (
    project_root_dir / "models" / "interpret" / "best_model_versions.yaml"
)
best_model_versions = OmegaConf.load(best_model_versions_path)

# Get specific run where SHAP explanation file is logged as an artifact
task_input = task_name.split("-")[0]
run = model_runs_df.loc[model_type, task_input]

# Get specific model name & corresponding SHAP explanation filename
model_name = f"{model_type}_{task_name}"
model_version = best_model_versions[model_type][task_input]

# Download SHAP explanation for model if does not exist locally
filename = f"{model_name}_v{model_version}_shap_explanation.pkl"
shap_artifact_path = Path("shap") / filename
local_dir_path = Path(__file__).parent
shap_local_path = Path(__file__).parent / shap_artifact_path
if not shap_local_path.exists():
    mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path=shap_artifact_path.as_posix(),
        dst_path=local_dir_path.as_posix(),
    )

# Load SHAP explanation
explanation = pickle.load(open(shap_local_path, "rb"))

# Get datamodule for reference to label & index mapping
datamodule = DataModule(project_dir=project_root_dir)
#%%
# Plot the Top Words impacting each specific class
class_index = 3
class_label = datamodule.asa_id2label[class_index]
mean_shap_explanation_for_class = explanation[:, :, class_index].mean(0)
print(f"Word features contributing to the most negative support for ASA {class_label}")
shap.plots.bar(mean_shap_explanation_for_class, order=shap.Explanation.argsort.flip)
print(
    f"Word features contributing to the most greatest positive support for ASA {class_label}"
)
shap.plots.bar(mean_shap_explanation_for_class, order=shap.Explanation.argsort)

# %%
