# %% [markdown]
# ## Compute all Metrics
# Run `evaluate_all_models` first to generate predictions for all models and all tasks.
# This script then computes all metrics and standard errors.
# Save results to pickle files.
# %%
from __future__ import annotations

import logging
import pickle
from pathlib import Path

from pytorch_lightning import seed_everything
from results_dataclasses import MetricResults
from tqdm.auto import tqdm

from src.metrics.functional import all_metrics

log = logging.getLogger(__name__)

seed_everything(seed=42, workers=True)

current_dir = Path(__file__).parent
# %%
# Load Model Predictions from Disk
model_predictions_path = current_dir / "model_predictions.pkl"
with open(model_predictions_path, "rb") as file:
    model_predictions = pickle.load(file)

# %%
# For All Tasks, Compute All Metrics
metric_results = {}
for task_input, task_predictions in (pbar1 := tqdm(model_predictions.items())):
    pbar1.set_description(f"Task Input: {task_input}")

    metric_results_path = current_dir / f"metric_results_{task_input}.pkl"
    try:
        # If existing pickle file for metrics, open and skip
        with open(metric_results_path, "rb") as file:
            metric_results[task_input] = pickle.load(file)
    except Exception:
        # Otherwise, compute metrics for task for all models
        class_labels = task_predictions.asa_class_names
        labels = task_predictions.labels
        num_classes = len(class_labels)
        metric_whitelist = [
            "Precision",
            "Recall",
            "F1",
            "AUROC",
            "AUPRC",
            "Precision/Micro",
            "Precision/Macro",
            "Recall/Micro",
            "Recall/Macro",
            "F1/Micro",
            "F1/Macro",
            "MCC/MCC",
            "ROC",
            "AUROC",
            "AUROC/Macro",
            "AUROC/AUCmu",
            "PrecisionRecallCurve",
            "AUPRC",
            "AUPRC/Macro",
        ]

        # Compute Metrics for Each Model
        task_metrics = {}
        model_names = [
            "rf",
            "svm",
            "fasttext",
            "bioclinicalbert",
            "age_meds",
            "random_classifier",
        ]
        for model_name in (pbar2 := tqdm(model_names)):
            pbar2.set_description(f"Metrics: {model_name}")
            preds = task_predictions.dict()[f"{model_name}_preds"]
            pred_proba = task_predictions.dict()[f"{model_name}_pred_proba"]
            if preds is not None:
                task_metrics[model_name] = all_metrics(
                    preds=preds,
                    pred_proba=pred_proba,
                    target=labels,
                    score_kind="probabilities",
                    include_loss=True,
                    num_classes=num_classes,
                    class_labels=class_labels,
                    whitelist=metric_whitelist,
                    bootstrap_std_error=True,
                )
            else:
                task_metrics[model_name] = None

        # Wrap Metric Results for Task in a DataClass
        task_metric_results = MetricResults(**task_metrics)
        metric_results[task_input] = task_metric_results
        # Save Metric Results for Task to Disk
        with open(metric_results_path, "wb") as file:
            pickle.dump(task_metric_results, file, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown]
# ## Test Statistical Significance for Pairwise Comparisons of Bootstrapped Metrics
# 1. enumerate all pairwise comparisons that exist for each metric
# 2. for each metric and condition (input task-model type combination), we have a vector
# of bootstrapped values
# 3. Compute p-values for all pairwise comparisons of different conditions & apply
# multiple hypothesis testing correction
# %%

import itertools

task_input = ["procedure", "diagnosis", "hpi", "ros", "pmsh", "meds", "note", "note512"]
model_names = ["rf", "svm", "fasttext", "bioclinicalbert"]
# Enumerate all input task-model type combinations
task_model_combination = list(itertools.product(task_input, model_names)) + [
    ("baseline", "age_meds"),
    ("baseline", "random_classifier"),
]

# Generate all unique pairwise comparisons
pairwise_comparison_names = []
for task_input1, model_name1 in task_model_combination:
    for task_input2, model_name2 in task_model_combination:
        # Accept pairs without same task & model name
        if task_input1 != task_input2 and model_name1 != model_name2:
            pairwise_comparison_names += [
                frozenset([(task_input1, model_name1), (task_input2, model_name2)])
            ]
# Uniquify duplicate pairs (e.g. (A, B) vs (B, A) = just one pairwise comparison)
pairwise_comparison_names = set(pairwise_comparison_names)
pairwise_comparison_names = [tuple(x) for x in pairwise_comparison_names]
print(f"There are {len(pairwise_comparison_names)} unique pairwise comparisons.")

# Now extract dictionary of flattened bootstrap metrics for each task-model-metric combination
task_model_boot_metrics = {}
for task_input, model_name in task_model_combination:
    # Get all metrics for task-model combination
    if task_input == "baseline":
        metric_dict = metric_results["note512"][model_name]
        # NOTE: baseline model metrics stored in "note512" MetricResults
        # as an implementation detail
    else:
        metric_dict = metric_results[task_input][model_name]
    # Drop all metrics except for bootstrap metrics
    boot_metric_dict = {
        k.removesuffix("_bootvalues"): v
        for k, v in metric_dict.items()
        if "_bootvalues" in k
    }
    # Flatten bootstrap metric hierarchy
    for metric_name, metric_value in boot_metric_dict.items():
        task_model_boot_metrics |= {(task_input, model_name, metric_name): metric_value}

# Get List of Metric Names
metric_names = []
for _, _, metric_name in task_model_boot_metrics.keys():
    if metric_name not in metric_names:
        metric_names += [metric_name]
print("Metric Names: ", metric_names)

# %%
# For each metric & for each of pairwise comparison, generate p-values by applying
# MannWhitneyU test to distributions generated from bootstraping each metric & experimental condition

import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats import multitest

pairwise_pvalue_for_metric = {}
for metric_name in (pbar1 := tqdm(metric_names)):
    pbar1.set_description(f"Pairwise p-values for {metric_name}")
    pairwise_pvalues = {}
    for pairwise_comparison_name in pairwise_comparison_names:
        (
            (task_input1, model_name1),
            (task_input2, model_name2),
        ) = pairwise_comparison_name

        distribution1 = task_model_boot_metrics[(task_input1, model_name1, metric_name)]
        distribution2 = task_model_boot_metrics[(task_input2, model_name2, metric_name)]

        metric_result = mannwhitneyu(x=distribution1, y=distribution2, method="auto")
        pvalue = metric_result.pvalue
        # Flip task-model name to model-task for formatting
        comparison_name = ((model_name1, task_input1), (model_name2, task_input2))
        pairwise_pvalues[comparison_name] = pvalue
    pairwise_pvalue_for_metric[metric_name] = pairwise_pvalues

pairwise_pvalue_for_metric_df = pd.DataFrame(pairwise_pvalue_for_metric).rename_axis(
    index=["Model-Task1", "Model-Task2"], columns=["Metric"]
)
pairwise_pvalue_for_metric_df
# %%
# Benjamini/Hochberg procedure for False Discovery Rate Control
alpha = 0.01

# Iterate through each metric (col) and apply correction to all pairwise comparisons for that metric
comparison_names = pairwise_pvalue_for_metric_df.index
reject_H0 = {}
corrected_pvalues = {}
for metric_name, col in pairwise_pvalue_for_metric_df.items():
    pvalues_for_metric = col
    reject, pvals_corrected, _, _ = multitest.multipletests(
        pvals=pvalues_for_metric, alpha=alpha, method="fdr_bh"
    )
    reject_H0[metric_name] = reject
    corrected_pvalues[metric_name] = pvals_corrected

reject_H0 = pd.DataFrame(reject_H0, index=comparison_names).rename_axis(
    index=["Model-Task1", "Model-Task2"], columns=["Metric"]
)
corrected_pvalues = pd.DataFrame(corrected_pvalues, index=comparison_names).rename_axis(
    index=["Model-Task1", "Model-Task2"], columns=["Metric"]
)

# Reshape corrected reject_H0 and p-values to list
reject_H0_list = (
    reject_H0.stack()
    .reorder_levels(["Metric", "Model-Task1", "Model-Task2"])
    .sort_index()
)
corrected_pvalues_list = (
    corrected_pvalues.stack()
    .reorder_levels(["Metric", "Model-Task1", "Model-Task2"])
    .sort_index()
)
# %%
# Get comparisons that are not statistically significant after Benjamini/Hochberg correction
not_statistically_significant = reject_H0_list[reject_H0_list.eq(False)]
# Get p-values for comparisons that are not statistically significant
not_statistically_significant_pvalues = corrected_pvalues_list.loc[
    not_statistically_significant.index
]
not_statistically_significant_pvalues = not_statistically_significant_pvalues.rename(
    "P-Value"
).to_frame()
not_statistically_significant_pvalues

# %%
# Get comparisons that are statistically significant
statistically_significant = reject_H0_list[reject_H0_list.eq(True)]
# Get p-values for comparisons that are statistically significant
statistically_significant_pvalues = corrected_pvalues_list.loc[
    statistically_significant.index
]
statistically_significant_pvalues = statistically_significant_pvalues.rename(
    "P-Value"
).to_frame()
# %%
# Export Tables
not_statistically_significant_pvalues.to_csv("pvalues_not_stat_sig.csv")
statistically_significant_pvalues.to_csv("pvalues_stat_sig.csv")

# %%
# Compare Differences between Original and B-H corrected P-values
orig_pvals = (
    pairwise_pvalue_for_metric_df.stack()
    .reorder_levels(["Metric", "Model-Task1", "Model-Task2"])
    .sort_index()
    .rename("Orig_P-Value")
    .to_frame()
)
corrected_pvals = pd.concat(
    [statistically_significant_pvalues, not_statistically_significant_pvalues]
).sort_index()

compare_pvals = orig_pvals.join(corrected_pvals).drop_duplicates()
compare_pvals
