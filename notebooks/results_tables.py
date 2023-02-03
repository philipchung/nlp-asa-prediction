#%% [markdown]
# # Results Tables
# This notebook shows all the results for each model type & task.
#
# Baseline Models:
# * __random classifier__: negative control baseline comparison model that randomly picks an ASA score without using any input features
# * __age classifier__: baseline comparison model that uses age to predict ASA score via simple logistic regression model
#
# Model Types:
# * __rf__: random forest, input featurized by unigram&bigram + TFIDF
# * __svm__: support vector machine, input featurized by unigram&bigram + TFIDF
# * __fasttext__: Facebook's fasttext model (similar to Word2Vec), input whitespace tokenized
# * __bioclinicalbert__: finetuned BioClinicalBERT model, input WordPiece tokenized
#
# Tasks: Predict ASA score (I, II, III, IV-V) given the following text snippet:
# * __procedure__: Description/name of planned surgery (or surgeries), extracted from Anesthetic Record.
# * __diagnosis__: Diagnosis at time of surgery booking, extracted from Anesthetic Record.
# * __hpi__: History of Present Illness. Narrative that summarizes patient's medical status and why they are having surgery.
# * __ros__: Review of Systems. Semi-structured narrative of medical conditions, issues, concerns, organized by organ systems
# * __pmsh__: Past Medical Surgical History. Combination of Past Medical History (list of medical conditions) and Past Surgical History (list of surgeries)
# * __meds__: List of medications patient is taking
# * __note__: Full Pre-Anesthesia note raw text from which the above sections have been extracted.  Also includes other note sections not described above.
# * __note512__: Pre-Anesthesia note truncated to first 512 WordPiece tokens (for equal comparison of other models to BioClinicalBERT which is limited to 512 tokens)
#%%
from __future__ import annotations

import copy
from typing import Any

import pandas as pd
from IPython.display import HTML, display
from models.interpret.mlflow_runs import get_baseline_runs, get_mlflow_runs

pd.options.display.float_format = "{:,.3f}".format

# Utility Functions


def flatten_dict(dictNested: dict[str, dict[str, Any]]) -> dict[tuple[str, str], Any]:
    "Converts 2-level nested dict to 1-level dict with tuple index."
    new_dict = {}
    for outerKey, innerDict in dictNested.items():
        for innerKey, values in innerDict.items():
            new_dict[(outerKey, innerKey)] = values
    return new_dict


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def separate_aggregate_classwise_metrics(
    metrics: dict, prefix: str = "", drop_prefix: bool = True
) -> dict[str, dict]:
    "Utility function to split run metrics into classwise and aggregate metrics."
    _metrics = copy.deepcopy(metrics)

    # Define Class-Aggregate Metric Keys
    aggregate_metric_names = [
        "MCC/MCC",
        "AUROC/AUCmu",
        "AUROC/Macro",
        "AUPRC/Macro",
        "F1/Micro",
        "F1/Macro",
        "F1/Weighted",
        "Precision/Micro",
        "Precision/Macro",
        "Precision/Weighted",
        "Recall/Micro",
        "Recall/Macro",
        "Recall/Weighted",
    ]
    if prefix:
        aggregate_metric_names = [
            f"{prefix}{metric_name}" for metric_name in aggregate_metric_names
        ]
    # Get Class-Aggregate Metrics
    aggregate_metrics = {}
    for name in aggregate_metric_names:
        new_name = remove_prefix(name, prefix) if drop_prefix else name
        aggregate_metrics[new_name] = _metrics[name]

    # Define Classwise Metrics Keys
    classwise_metric_names = ["AUROC", "AUPRC", "F1", "Precision", "Recall"]
    asa_classes = ["I", "II", "III", "IV-V"]
    if prefix:
        classwise_metric_names = [
            f"{prefix}{metric_name}" for metric_name in classwise_metric_names
        ]
    # Get Classwise Metric Keys
    classwise_metrics = {}
    for name in classwise_metric_names:
        new_name = remove_prefix(name, prefix) if drop_prefix else name
        classwise_metrics[new_name] = {
            f"{asa_class}": _metrics[f"{name}/{asa_class}"] for asa_class in asa_classes
        }

    return {
        "aggregate": aggregate_metrics,
        "classwise": classwise_metrics,
    }


#%% [markdown]
# ## Baseline Model Performance
#%%
# Get Baseline runs
baselines = get_baseline_runs()
random_baseline_run = baselines["random_classifier"]
age_baseline_run = baselines["age_classifier"]
# Split metrics into Class-aggregate and Class-specific Metrics
random_baseline_metrics = separate_aggregate_classwise_metrics(
    random_baseline_run.data.metrics, prefix="ASA/Test/"
)
age_baseline_metrics = separate_aggregate_classwise_metrics(
    age_baseline_run.data.metrics, prefix="ASA/Test/"
)
#%%
# Get Info on Metrics Available & Number of Samples in each ASA Class
metric_dict = random_baseline_run.data.metrics
print(f"List of Metrics Available: {list(metric_dict.keys())}")
print("\n")
print("Support (Number of Samples) for Each ASA Class in Test Set")
support_dict = {
    "I": metric_dict["ASA/Test/Support/I"],
    "II": metric_dict["ASA/Test/Support/II"],
    "III": metric_dict["ASA/Test/Support/III"],
    "IV-V": metric_dict["ASA/Test/Support/IV-V"],
}
num_samples = pd.Series(support_dict).astype(int).rename("Num Samples")
fraction_samples = (num_samples / num_samples.sum()).rename("Fraction")
pd.concat([num_samples, fraction_samples], axis=1)

#%%
# Class-aggregate metrics table for baseline models
aggregate_baseline_df = pd.DataFrame.from_dict(
    {
        "Random Classifier": random_baseline_metrics["aggregate"],
        "Age Classifier": age_baseline_metrics["aggregate"],
    },
    orient="columns",
).T
print("Class-Aggregate Baseline Metrics")
aggregate_baseline_df

#%%
# Class-specific metrics table for baseline models
classwise_baseline_dict = {
    "Random Classifier": flatten_dict(random_baseline_metrics["classwise"]),
    "Age Classifier": flatten_dict(age_baseline_metrics["classwise"]),
}
classwise_baseline_df = pd.DataFrame.from_dict(
    classwise_baseline_dict, orient="index"
).stack()
print("Class-Specific Baseline Metrics")
classwise_baseline_df
#%% [markdown]
# ## Experimental Model & Task-specific Performance
#%%
# Get MLflow runs for RF, SVM, fastText, BioClinicalBERT
runs_dict = get_mlflow_runs(child_run_types="test_runs")
test_runs_df = runs_dict["test_runs"]
test_runs_df = test_runs_df[
    ["diagnosis", "procedure", "hpi", "pmsh", "ros", "meds", "note", "note512"]
].rename(
    columns={
        "diagnosis": "Diagnosis",
        "procedure": "Procedure",
        "hpi": "HPI",
        "pmsh": "PMSH",
        "ros": "ROS",
        "meds": "Meds",
        "note": "Note",
        "note512": "Note512",
    },
    index={
        "rf": "Random Forest",
        "svm": "Support Vector Machine",
        "fasttext": "fastText",
        "bioclinicalbert": "BioClinicalBERT",
    },
)
# Split metrics into Class-aggregate and Class-specific Metrics
test_run_metrics_df = test_runs_df.applymap(
    lambda run: separate_aggregate_classwise_metrics(
        run.data.metrics, prefix="ASA/Test/"
    )
)
#%% [markdown]
# ### Matthew's Correlation Coefficient (MCC)
# %%
print("Matthew's Correlation Coefficient (MCC)")
mcc_df = test_run_metrics_df.applymap(lambda d: d["aggregate"]["MCC/MCC"])
mcc_df
#%% [markdown]
# ### AUCmu (Multiclass U-Statistic Generalization of AUROC")
# %%
print("AUCmu (Multiclass U-Statistic Generalization of AUROC")
aucmu_df = test_run_metrics_df.applymap(lambda d: d["aggregate"]["AUROC/AUCmu"])
aucmu_df

#%% [markdown]
# ### Area Under Receiver Operator Characteristic Curve (AUROC)
# %%
print("Macro-average AUROC")
auroc_macro_df = test_run_metrics_df.applymap(lambda d: d["aggregate"]["AUROC/Macro"])
auroc_macro_df
#%%
print("Class-specific AUROC")
classwise_auroc_df = test_run_metrics_df.applymap(lambda d: d["classwise"]["AUROC"])
classwise_auroc_df = classwise_auroc_df.stack().apply(pd.Series).stack().unstack(-2)
classwise_auroc_df


#%% [markdown]
# ### Area Under Precision-Recall Curve (AUPRC)
# %%
print("Macro-average AUPRC")
auprc_macro_df = test_run_metrics_df.applymap(lambda d: d["aggregate"]["AUPRC/Macro"])
auprc_macro_df
# %%
print("Class-specific AUPRC")
classwise_auprc_df = test_run_metrics_df.applymap(lambda d: d["classwise"]["AUPRC"])
classwise_auprc_df = classwise_auprc_df.stack().apply(pd.Series).stack().unstack(-2)
classwise_auprc_df

#%% [markdown]
# ### F1 Score (Harmonic Mean of Precision & Recall)
# %%
print("Macro-average F1")
f1_macro_df = test_run_metrics_df.applymap(lambda d: d["aggregate"]["F1/Macro"])
f1_macro_df
#%%
print("Class-specific F1")
classwise_f1_df = test_run_metrics_df.applymap(lambda d: d["classwise"]["F1"])
classwise_f1_df = classwise_f1_df.stack().apply(pd.Series).stack().unstack(-2)
classwise_f1_df

#%% [markdown]
# ### Precision
#%%
print("Macro-average Precision")
precision_macro_df = test_run_metrics_df.applymap(
    lambda d: d["aggregate"]["Precision/Macro"]
)
precision_macro_df
#%%
print("Class-specific Precision")
classwise_precision_df = test_run_metrics_df.applymap(
    lambda d: d["classwise"]["Precision"]
)
classwise_precision_df = (
    classwise_precision_df.stack().apply(pd.Series).stack().unstack(-2)
)
classwise_precision_df

#%% [markdown]
# ### Recall
#%%
print("Macro-average Recall")
recall_macro_df = test_run_metrics_df.applymap(lambda d: d["aggregate"]["Recall/Macro"])
recall_macro_df
#%%
print("Class-specific Recall")
classwise_recall_df = test_run_metrics_df.applymap(lambda d: d["classwise"]["Recall"])
classwise_recall_df = classwise_recall_df.stack().apply(pd.Series).stack().unstack(-2)
classwise_recall_df
#%%
