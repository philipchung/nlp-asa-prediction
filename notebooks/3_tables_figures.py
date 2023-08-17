# %% [markdown]
# ## Compute all Metrics
# Run `evaluate_all_models` first to generate predictions for all models and all tasks.
# This script then computes all metrics and standard errors.
# Save results to pickle file.
# %%
from __future__ import annotations

import logging
import pickle
from functools import partial
from pathlib import Path

import pandas as pd
import torch
from pytorch_lightning import seed_everything
from results_dataclasses import MetricResults, ModelPredictions
from tqdm.auto import tqdm

from src.utils import flatten_list_of_list

log = logging.getLogger(__name__)

seed_everything(seed=42, workers=True)

current_dir = Path(__file__).parent

# Dummy Placeholder
metric_results = MetricResults()
model_predictions = ModelPredictions()
# %%
# Load Model Predictions from Disk
model_predictions_path = current_dir / "model_predictions.pkl"
with open(model_predictions_path, "rb") as file:
    model_predictions = pickle.load(file)

note512_predictions = model_predictions["note512"]

# Get Info on Number of Samples in each ASA Class
asa_id2label = {v: k for k, v in note512_predictions.asa_label2id.items()}
num_samples = note512_predictions.labels.shape[0]
asa_dist = (
    pd.Series(data=note512_predictions.labels, name="ASA Count")
    .value_counts()
    .sort_index(ascending=True)
)
asa_dist.index = asa_dist.index.to_series().apply(lambda x: asa_id2label[x])
asa_dist_fraction = (asa_dist / num_samples).rename("ASA Fraction")
pd.concat([asa_dist, asa_dist_fraction], axis=1)
# %%
# Load Metrics For All Tasks & Models
metric_results = {}
for task_input, task_predictions in (pbar1 := tqdm(model_predictions.items())):
    metric_results_path = current_dir / f"metric_results_{task_input}.pkl"
    with open(metric_results_path, "rb") as file:
        metric_results[task_input] = pickle.load(file)

metric_results.keys()
# %%
metric_results["note512"].keys()

# %%
# Utility functions


def remove_prefix(text, prefix) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def separate_aggregate_classwise_metrics(
    metrics: dict,
    prefix: str = "",
    drop_prefix: bool = True,
    aggregate_metric_names: list[str] = [
        "MCC/MCC",
        "AUROC/AUCmu",
        "AUROC/Macro",
        "AUPRC/Macro",
        "F1/Micro",
        "F1/Macro",
        "Precision/Micro",
        "Precision/Macro",
        "Recall/Micro",
        "Recall/Macro",
    ],
    classwise_metric_names: list[str] = ["AUROC", "AUPRC", "F1", "Precision", "Recall"],
    asa_classes: list[str] = ["I", "II", "III", "IV-V"],
) -> dict[str, dict]:
    "Utility function to split run metrics into classwise and aggregate metrics."
    # Refine Class-Aggregate Metric Keys
    if prefix:
        aggregate_metric_names = [
            f"{prefix}{metric_name}" for metric_name in aggregate_metric_names
        ]
    aggregate_metric_se_names = [
        f"{metric_name}_SE" for metric_name in aggregate_metric_names
    ]
    # Refine Classwise Metrics Keys
    if prefix:
        classwise_metric_names = [
            f"{prefix}{metric_name}" for metric_name in classwise_metric_names
        ]
    expanded_classwise_metric_names = flatten_list_of_list(
        [f"{metric_name}/{asa_class}" for asa_class in asa_classes]
        for metric_name in classwise_metric_names
    )
    expanded_classwise_metric_se_names = [
        f"{metric_name}_SE" for metric_name in expanded_classwise_metric_names
    ]

    # Separate Metrics
    aggregate_metrics = {}
    aggregate_metrics_se = {}
    classwise_metrics = {}
    classwise_metrics_se = {}
    for metric_name, metric_value in metrics.items():
        metric_name = remove_prefix(metric_name, prefix) if drop_prefix else metric_name
        # Convert 1-element Tensors to Python Scalars
        if isinstance(metric_value, torch.Tensor):
            if metric_value.shape == ():
                metric_value = metric_value.item()
        if metric_name in aggregate_metric_names:
            aggregate_metrics[metric_name] = metric_value
        elif metric_name in aggregate_metric_se_names:
            name = metric_name.rstrip("_SE")
            aggregate_metrics_se[name] = metric_value
        elif metric_name in expanded_classwise_metric_names:
            classwise_metrics[metric_name] = metric_value
        elif metric_name in expanded_classwise_metric_se_names:
            name = metric_name.rstrip("_SE")
            classwise_metrics_se[name] = metric_value
        else:
            log.warning(
                f"Skipping metric: {metric_name} because it is "
                "not in aggregate or classwise metric collections."
            )
    return {
        "aggregate": aggregate_metrics,
        "aggregate_se": aggregate_metrics_se,
        "classwise": classwise_metrics,
        "classwise_se": classwise_metrics_se,
    }


def filter_classwise(d: dict, prefix: str) -> dict:
    "Filter metric dict by metric prefix, then returns classwise dict for that metric."
    output = {}
    for k, v in d.items():
        if prefix in k:
            new_key = k.split("/")[1]
            output |= {new_key: v}
    return output


def columnwise_str_format(
    col_from_df1: pd.Series, col_from_df2: pd.Series
) -> pd.Series:
    "Use with pd.DataFrame.combine to format metrics with standard error."
    combined = []
    for item1, item2 in zip(col_from_df1, col_from_df2):
        text = f"{item1:,.3f} ({item2:.0E})"
        combined += [text]
    return pd.Series(data=combined, index=col_from_df1.index, name=col_from_df1.name)


# %%
# Unpack all nested result objects into a metrics dataframe
metrics = {}
baseline_metrics = {}
for task_input, task_metric_results_obj in (pbar1 := tqdm(metric_results.items())):
    for model_type, task_metric_results in (
        pbar2 := tqdm(task_metric_results_obj.dict().items())
    ):
        if model_type in ("rf", "svm", "fasttext", "bioclinicalbert"):
            metrics |= {
                (task_input, model_type): separate_aggregate_classwise_metrics(
                    task_metric_results
                )
            }
        elif model_type in ("age_meds", "random_classifier"):
            if task_metric_results:
                baseline_metrics |= {
                    (task_input, model_type): separate_aggregate_classwise_metrics(
                        task_metric_results
                    )
                }

# %%
metrics_df = pd.DataFrame(metrics).stack(1)
baseline_metrics_df = pd.DataFrame(baseline_metrics)
baseline_metrics_df.columns = baseline_metrics_df.columns.droplevel(0)
# Split "aggregate", "aggregate_se", "classwise", "classwise_se" dimension into
# 4 separate dataframes.
baseline_metrics_aggregate = baseline_metrics_df.loc["aggregate"]
baseline_metrics_aggregate_se = baseline_metrics_df.loc["aggregate_se"]
baseline_metrics_classwise = baseline_metrics_df.loc["classwise"]
baseline_metrics_classwise_se = baseline_metrics_df.loc["classwise_se"]
# Split "aggregate", "aggregate_se", "classwise", "classwise_se" dimension into
# 4 separate dataframes.
metrics_aggregate = metrics_df.loc[("aggregate", slice(None))]
metrics_aggregate_se = metrics_df.loc[("aggregate_se", slice(None))]
metrics_classwise = metrics_df.loc[("classwise", slice(None))]
metrics_classwise_se = metrics_df.loc[("classwise_se", slice(None))]

# NOTE: each resultant dataframe cell contains a dictionary of metrics.  To get a specific
# metric, we can applymap over the entire dataframe and access just a specific dictionary item
# %%
# Reorder Row & Columns
model_order = ["rf", "svm", "fasttext", "bioclinicalbert"]
task_order = ["diagnosis", "procedure", "hpi", "pmsh", "ros", "meds", "note", "note512"]
metrics_aggregate = metrics_aggregate.reindex(index=model_order, columns=task_order)
metrics_aggregate_se = metrics_aggregate_se.reindex(
    index=model_order, columns=task_order
)
metrics_classwise = metrics_classwise.reindex(index=model_order, columns=task_order)
metrics_classwise_se = metrics_classwise_se.reindex(
    index=model_order, columns=task_order
)
# Format Row & Column Names
model_name_map = {
    "rf": "Random Forest",
    "svm": "Support Vector Machine",
    "fasttext": "fastText",
    "bioclinicalbert": "BioClinicalBERT",
    "age_meds": "Age & Meds Classifier",
    "random_classifier": "Random Classifier",
}
task_name_map = {
    "diagnosis": "Diagnosis",
    "procedure": "Procedure",
    "hpi": "HPI",
    "pmsh": "PMSH",
    "ros": "ROS",
    "meds": "Meds",
    "note": "Note",
    "note512": "Note512",
}
# %% [markdown]
# ### Baseline Metrics
# %%
print("Baseline Class-Aggregate Metrics: Age Meds & Random Classifier")
baseline_agg = baseline_metrics_aggregate.apply(pd.Series)
baseline_agg_se = baseline_metrics_aggregate_se.apply(pd.Series)
baseline_agg.combine(baseline_agg_se, func=columnwise_str_format)
# %%
print("Baseline Class-Specific Metrics: Age Meds & Random Classifier")
baseline_classwise = baseline_metrics_classwise.apply(pd.Series)
baseline_classwise.columns = pd.MultiIndex.from_tuples(
    [x.split("/") for x in baseline_classwise.columns.to_list()]
)
baseline_classwise = baseline_classwise.stack()
baseline_classwise_se = baseline_metrics_classwise_se.apply(pd.Series)
baseline_classwise_se.columns = pd.MultiIndex.from_tuples(
    [x.split("/") for x in baseline_classwise_se.columns.to_list()]
)
baseline_classwise_se = baseline_classwise_se.stack()
baseline_classwise.combine(baseline_classwise_se, func=columnwise_str_format)

# %% [markdown]
# ### Matthew's Correlation Coefficient (MCC)
# %%
print("Matthew's Correlation Coefficient (MCC)")
mcc = metrics_aggregate.applymap(lambda d: d["MCC/MCC"])
mcc_se = metrics_aggregate_se.applymap(lambda d: d["MCC/MCC"])
mcc.combine(mcc_se, func=columnwise_str_format)
# %% [markdown]
# ### AUCmu (Multiclass U-Statistic Generalization of AUROC")
# %%
print("AUCmu (Multiclass U-Statistic Generalization of AUROC")
aucmu = metrics_aggregate.applymap(lambda d: d["AUROC/AUCmu"])
aucmu_se = metrics_aggregate_se.applymap(lambda d: d["AUROC/AUCmu"])
aucmu.combine(aucmu_se, func=columnwise_str_format)
# %% [markdown]
# ### Area Under Receiver Operator Characteristic Curve (AUROC)
# %%
print("Macro-average AUROC")
auroc_macro = metrics_aggregate.applymap(lambda d: d["AUROC/Macro"])
auroc_macro_se = metrics_aggregate_se.applymap(lambda d: d["AUROC/Macro"])
auroc_macro.combine(auroc_macro_se, func=columnwise_str_format)
# %%
print("Class-specific AUROC")
fn = partial(filter_classwise, prefix="AUROC")
classwise_auroc = (
    metrics_classwise.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_auroc_se = (
    metrics_classwise_se.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_auroc.combine(classwise_auroc_se, func=columnwise_str_format)
# %% [markdown]
# ### Area Under Precision-Recall Curve (AUPRC)
# %%
print("Macro-average AUPRC")
auprc_macro = metrics_aggregate.applymap(lambda d: d["AUPRC/Macro"])
auprc_macro_se = metrics_aggregate_se.applymap(lambda d: d["AUPRC/Macro"])
auprc_macro.combine(auprc_macro_se, func=columnwise_str_format)
# %%
print("Class-specific AUPRC")
fn = partial(filter_classwise, prefix="AUPRC")
classwise_auprc = (
    metrics_classwise.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_auprc_se = (
    metrics_classwise_se.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_auprc.combine(classwise_auprc_se, func=columnwise_str_format)
# %% [markdown]
# ### F1 Score (Harmonic Mean of Precision & Recall)
# %%
print("Macro-average F1")
f1_macro = metrics_aggregate.applymap(lambda d: d["F1/Macro"])
f1_macro_se = metrics_aggregate_se.applymap(lambda d: d["F1/Macro"])
f1_macro.combine(f1_macro_se, func=columnwise_str_format)
# %%
print("Class-specific F1")
fn = partial(filter_classwise, prefix="F1")
classwise_f1 = (
    metrics_classwise.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_f1_se = (
    metrics_classwise_se.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_f1.combine(classwise_f1_se, func=columnwise_str_format)
# %% [markdown]
# ### Precision
# %%
print("Macro-average Precision")
precision_macro = metrics_aggregate.applymap(lambda d: d["Precision/Macro"])
precision_macro_se = metrics_aggregate_se.applymap(lambda d: d["Precision/Macro"])
precision_macro.combine(precision_macro_se, func=columnwise_str_format)
# %%
print("Class-specific Precision")
fn = partial(filter_classwise, prefix="Precision")
classwise_precision = (
    metrics_classwise.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_precision_se = (
    metrics_classwise_se.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_precision.combine(classwise_precision_se, func=columnwise_str_format)
# %% [markdown]
# ### Recall
# %%
print("Macro-average Recall")
recall_macro = metrics_aggregate.applymap(lambda d: d["Recall/Macro"])
recall_macro_se = metrics_aggregate_se.applymap(lambda d: d["Recall/Macro"])
recall_macro.combine(recall_macro_se, func=columnwise_str_format)
# %%
print("Class-specific Recall")
fn = partial(filter_classwise, prefix="Recall")
classwise_recall = (
    metrics_classwise.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_recall_se = (
    metrics_classwise_se.applymap(fn).stack().apply(pd.Series).stack().unstack(-2)
)
classwise_recall.combine(classwise_recall_se, func=columnwise_str_format)
# %%
