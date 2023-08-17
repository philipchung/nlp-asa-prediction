# %%
from __future__ import annotations

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from models.utils import prepare_datamodule, resolve_paths_and_save_config
from src.dataset.data import Data
from src.metrics.format import format_all
from src.metrics.functional import all_metrics
from src.metrics.utils import arraylike_to_tensor

az_http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
az_http_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)
prn_handling = "Categorical"
output_label_name = "asa_label"
model = None
features = "AgeMedsCountRxNorm"

project_dir = Path("/home/azureuser/cloudfiles/code/Users/chungph/presurgnlp-az")
data_dir = project_dir / "data/id/v3"
meds_data_path = data_dir / "raw/meds_prior_to_start.tsv"


with hydra.initialize(config_path="../conf", version_base=None):
    cfg = hydra.compose(config_name="baseline_config")
    cfg = resolve_paths_and_save_config(cfg, file_ref=__file__)
datamodule = prepare_datamodule(cfg=cfg)

# Load Meds Data
# This is the active meds list at time of AnesStart.  Criteria:
# - CURRENT_START_DT_TM < AnesStart
# - AnesStart < PROJECTED_STOP_DT_TM or DISCONTINUE_EFFECTIVE_DT_TM
meds_df = pd.read_csv(
    meds_data_path,
    sep="\t",
    encoding="latin-1",
)

# Mapping Between Medication Name -> RxNorm
rxnorm_mapping = (
    meds_df.loc[:, ["MedicationName", "RxNormCode"]]
    .dropna()
    .drop_duplicates()
    .set_index("MedicationName")
    .astype(int)
    .squeeze()
)
# Some Med names have slightly different spelling/capitalization,
# so we drop the different spelling variants to have a 1:1 mapping.
rxnorm_mapping = rxnorm_mapping.drop_duplicates()
name2rxnorm = rxnorm_mapping.to_dict()
rxnorm2name = {v: k for k, v in name2rxnorm.items()}

# Drop Meds w/o RxNormCode
meds_df = meds_df.loc[meds_df.RxNormCode.notna()]
meds_df.RxNormCode = meds_df.RxNormCode.astype(int)

# Identify if Med is PRN or not
is_prn = meds_df.CLINICAL_DISPLAY_LINE.apply(lambda text: "PRN" in str(text))
# Define whether PRN Meds are Featurized differently then Non-PRN Meds
# NOTE: Meds not prescribed are not present in `meds_df` and will be set to 0
# for each proc_id if not present when we apply pivot_table
if prn_handling == "Categorical":
    # Treat Med & PRN status as Categorical
    # Values: 0=Not Prescribed, 1= Prescribed PRN Med, 2=Prescribed Non-PRN Med
    value = is_prn.apply(lambda x: 1 if x else 2)
elif prn_handling == "Drop":
    # Drop PRN Meds from being used as Features
    # Values: 0=Not Prescribed or PRN Med, 1=Prescribed Med
    value = is_prn.apply(lambda x: 0 if x else 1)
elif prn_handling == "Full":
    # Treat PRN Meds and Non-PRN Meds the same and include both as Features
    # Values: 0=Not Prescribed, 1=Prescribed Non-PRN Med or PRN Med
    value = 1
else:
    raise ValueError("Unknown value for argument `prn_handling`.")

# Get Table of Med Presence for each ProcID
meds_df = meds_df.assign(Present=1, IsPRN=is_prn, Value=value)
meds_for_proc_id = pd.pivot_table(
    data=meds_df,
    index="ProcedureID",
    columns="RxNormCode",
    values="Value",
    fill_value=0,
).astype(int)
# Get Reference of Med Feature Columns
meds_feature_rxnorm = meds_for_proc_id.columns.tolist()
# Total Number of Unique Meds Per ProcID
meds_count = meds_for_proc_id.sum(axis=1).rename("MedsCount")

# Get Age from Cases Table
data = Data(
    project_dir=datamodule.project_dir,
    datastore_blob_path=datamodule.data_blob_prefix,
    dataset_name=datamodule.dataset_name,
    seed=datamodule.seed,
)
cases = data.get_cases()

# Define Data Splits, Join Age from Cases Table
columns = [
    "ProcedureID",
    "ASA",
    "asa_class",
    "emergency",
    "asa_label",
    "emergency_label",
]
train_df = (
    datamodule.train_dataframe(columns=columns)
    .reset_index()
    .set_index("ProcedureID")
    .join(cases.Age)
    .join(meds_count)
    .join(meds_for_proc_id)
    .fillna(False)
    .reset_index()
    .set_index("index")
)
val_df = (
    datamodule.val_dataframe(columns=columns)
    .reset_index()
    .set_index("ProcedureID")
    .join(cases.Age)
    .join(meds_count)
    .join(meds_for_proc_id)
    .fillna(False)
    .reset_index()
    .set_index("index")
)
test_df = (
    datamodule.test_dataframe(columns=columns)
    .reset_index()
    .set_index("ProcedureID")
    .join(cases.Age)
    .join(meds_count)
    .join(meds_for_proc_id)
    .fillna(False)
    .reset_index()
    .set_index("index")
)

# Define input & output data
if features == "Age":
    feature_columns = ["Age"]
elif features == "AgeMedsCount":
    feature_columns = ["Age", "MedsCount"]
elif features == "AgeRxNorm":
    feature_columns = ["Age"] + meds_feature_rxnorm
elif features == "MedsCountRxNorm":
    feature_columns = ["MedsCount"] + meds_feature_rxnorm
elif features == "RxNorm":
    feature_columns = meds_feature_rxnorm
elif features == "AgeMedsCountRxNorm":
    feature_columns = ["Age", "MedsCount"] + meds_feature_rxnorm
else:
    raise ValueError("Unknown input for `feature`.")

X_train = train_df.loc[:, feature_columns].to_numpy(dtype=np.int64)
y_train = train_df[output_label_name].to_numpy(dtype=np.int64)
X_val = val_df.loc[:, feature_columns].to_numpy(dtype=np.int64)
y_val = val_df[output_label_name].to_numpy(dtype=np.int64)
X_test = test_df.loc[:, feature_columns].to_numpy(dtype=np.int64)
y_test = test_df[output_label_name].to_numpy(dtype=np.int64)

# Default Model if None Specified
if model is None:
    model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
        max_iter=10000,
        multi_class="multinomial",
        n_jobs=-1,
    )

# Fit Model
model.fit(X_train, y_train)

# Make Predictions on Val Dataset
y_val_pred = model.predict(X_val)
y_val_pred_proba = model.predict_proba(X_val)

# Make Predictions on Test Dataset
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)

# %%
# Test Bootstrap Metrics - proof of concept on classwise metrics
preds = arraylike_to_tensor(y_test_pred)
pred_proba = arraylike_to_tensor(y_test_pred_proba)
target = arraylike_to_tensor(y_test)

num_classes = 4
class_labels = datamodule.asa_class_names

test_metrics = all_metrics(
    preds=y_test_pred,
    pred_proba=y_test_pred_proba,
    target=y_test,
    score_kind="probabilities",
    include_loss=True,
    num_classes=datamodule.asa_num_classes,
    class_labels=datamodule.asa_class_names,
    top_k_easiest_hardest=cfg.metrics.examples_top_k,
    dataset=datamodule.test,
    id2label=datamodule.asa_id2label,
    whitelist=cfg.metrics.whitelist,
    # bootstrap_std_error=True,
    bootstrap_std_error=False,
)
test_metrics
# %%
test_metrics = format_all(
    metrics=test_metrics,
    class_labels=datamodule.asa_class_names,
    split="Test",
    target_name="ASA",
)

# %%
from azureml.core import Workspace
import mlflow

ws = Workspace.from_config()
mlflow_uri = ws.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(mlflow_uri)
tags = {"ModelType": "AgeMedsClassifier"}

experiment_id = mlflow.create_experiment(
    "Debug",
    tags=tags,
)
with mlflow.start_run(
    experiment_id=experiment_id,
    description="debugging",
):
    project_root = datamodule.project_dir.resolve()
    pip_requirements_path = (project_root / "requirements.txt").as_posix()
    code_paths = [Path(__file__).parent.as_posix()]
    registered_model_name = f"age_meds_classifier_{cfg.task.task_name}"
    # MLModel Signature
    input_name = cfg.model.data.input_feature_name
    output_name = cfg.model.data.output_label_name
    input_data_sample = pd.DataFrame(X_train[:5,:], columns=feature_columns)
    output_data_sample = pd.DataFrame(y_train[:5], columns=[output_name])
    signature = mlflow.models.infer_signature(
        input_data_sample, output_data_sample
    )
    # Log Model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        pip_requirements=pip_requirements_path,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_data_sample,
    )
# %%