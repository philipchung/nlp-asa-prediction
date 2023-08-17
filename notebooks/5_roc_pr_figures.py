# %% [markdown]
# ## ROC & PR Curve Figures
# Generate ROC and PR figures.  Run `evaluate_all_models.py` script first.
# %%
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns
import torch
from pytorch_lightning import seed_everything
from results_dataclasses import MetricResults, ModelPredictions
from torchmetrics.functional.classification import (
    multiclass_precision_recall_curve,
    multiclass_roc,
)
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

seed_everything(seed=42, workers=True)

current_dir = Path(__file__).parent

# Dummy Placeholder
metric_results = MetricResults()
model_predictions = ModelPredictions()
# %% [markdown]
# ## Load Results of Model Evaluation
# %%
# Load Model Predictions from Disk
model_predictions_path = current_dir / "model_predictions.pkl"
with open(model_predictions_path, "rb") as file:
    model_predictions = pickle.load(file)

note512_predictions = model_predictions["note512"]
class_labels = note512_predictions.asa_class_names

# Compute ROC & PrecisionRecallCurve
num_thresholds = 200
metric_results = {}
for task_input, task_predictions in (pbar1 := tqdm(model_predictions.items())):
    pbar1.set_description(f"Task Input: {task_input}")

    # Otherwise, compute metrics for task for all models
    class_labels = task_predictions.asa_class_names
    labels = task_predictions.labels
    num_classes = len(class_labels)
    metric_whitelist = ["ROC", "PrecisionRecallCurve"]

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
        pred_proba = task_predictions.dict()[f"{model_name}_pred_proba"]
        if pred_proba is not None:
            roc = multiclass_roc(
                preds=torch.from_numpy(pred_proba),
                target=torch.from_numpy(labels),
                num_classes=num_classes,
                thresholds=num_thresholds,
            )
            pr_curve = multiclass_precision_recall_curve(
                preds=torch.from_numpy(pred_proba),
                target=torch.from_numpy(labels),
                num_classes=num_classes,
                thresholds=num_thresholds,
            )
            task_metrics[model_name] = {"ROC": roc, "PrecisionRecallCurve": pr_curve}
        else:
            task_metrics[model_name] = None

    # Wrap Metric Results for Task in a DataClass
    task_metric_results = MetricResults(**task_metrics)
    metric_results[task_input] = task_metric_results


# %% [markdown]
# ## Compute ROC Curve & Precision Recall Curve Data
# %%
def reformat_roc(roc: torch.Tensor, class_names: tuple | list) -> pd.DataFrame:
    # Convert Tuple of Tensors into Dict of Tensors
    roc_dict = dict(
        zip(
            ["fpr", "tpr", "thresholds"],
            roc,
        )
    )
    # Extract ROC by Class
    roc_data = []
    for idx, class_name in enumerate(class_names):
        class_df = pd.DataFrame.from_dict(
            {"FPR": roc_dict["fpr"][idx], "TPR": roc_dict["tpr"][idx]}
        )
        class_df["ASA"] = class_name
        roc_data += [class_df]
    # Combine ROC from all Classes into single long dataframe
    roc_df = pd.concat(roc_data, axis=0)
    return roc_df


def reformat_pr(pr: torch.Tensor, class_names: tuple | list) -> pd.DataFrame:
    # Convert Tuple of Tensors into Dict of Tensors
    pr_dict = dict(zip(["precision", "recall", "thresholds"], pr))
    # Extract PrecisionRecallCurve
    pr_data = []
    for idx, class_name in enumerate(class_names):
        class_df = pd.DataFrame.from_dict(
            {"Precision": pr_dict["precision"][idx], "Recall": pr_dict["recall"][idx]}
        )
        class_df["ASA"] = class_name
        pr_data += [class_df]
    # Combine PrecisionRecallCurve from all Classes into single long dataframe
    pr_df = pd.concat(pr_data, axis=0)
    return pr_df


def force_precision_to_class_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Corrects random classifier PR curve to be a flat horizontal line corresponding
    to the class's frequency.
    In torchmetrics, when computing PR curves, initial Precision always starts at 0.0 or 1.0.
    But will result in a non-truthful curve.
    """
    precision_class_frequency = df.loc[df.Recall == 1.0].Precision.iloc[0]
    df.Precision = precision_class_frequency
    return df


# Get ROC & PrecisionRecallCurve for All Models for Task=Note512
# and Reformat into DataFrame Long Format
note512_metric_results = metric_results["note512"]
name_map = {
    "rf": "Random Forest",
    "svm": "Support Vector Machine",
    "fasttext": "fastText",
    "bioclinicalbert": "BioClinicalBERT",
    "age_meds": "Age & Meds",
    "random_classifier": "Random Classifier",
}
roc_df_list = []
pr_df_list = []
for model_type in note512_metric_results.dict().keys():
    roc = note512_metric_results.dict()[model_type]["ROC"]
    roc_df = reformat_roc(roc=roc, class_names=class_labels)
    roc_df = roc_df.assign(Model=name_map[model_type])
    roc_df_list += [roc_df]

    pr = note512_metric_results.dict()[model_type]["PrecisionRecallCurve"]
    pr_df = reformat_pr(pr=pr, class_names=class_labels)
    pr_df = pr_df.assign(Model=name_map[model_type])
    if model_type == "random_classifier":
        pr_df = pr_df.groupby("ASA", group_keys=False).apply(
            force_precision_to_class_frequency
        )

    pr_df_list += [pr_df]

# Combine All ROC Data into Single DataFrame
roc_df = pd.concat(roc_df_list, axis=0)
pr_df = pd.concat(pr_df_list, axis=0)

# TODO: fix pr_df data to make precision recall curve straight

# %% [markdown]
# ## Receiver Operator Characteristic Curve Figure
# %%
# Facet Grid (like a subplot) of ROC Curve for all ASA-PS for all 6 models
sns.set_style("whitegrid")
sns.plotting_context("poster")
sns.color_palette("colorblind")
p1 = sns.relplot(
    kind="line",
    data=roc_df,
    x="FPR",
    y="TPR",
    hue="Model",
    style="Model",
    errorbar=None,
    col="ASA",
    col_wrap=2,
    dashes={
        "Random Forest": (4, 4),
        "Support Vector Machine": (3, 3),
        "fastText": (2, 2),
        "BioClinicalBERT": (1, 1),
        "Age & Meds": (8, 8),
        "Random Classifier": (8, 8),
    },
    facet_kws={
        "sharex": True,
        "sharey": True,
        "legend_out": True,
        "despine": False,
    },
)

# Reformat Subplot Titles to be "ASA x"
p1.set_titles(template="{col_var} {col_name}")

# Overall Figure Title
sup_title = p1.fig.suptitle(
    "Receiver Operator Characteristic Curves for Note512 Task", fontsize=16
)
# Share x-label & y-label
sup_xlabel = p1.fig.supxlabel("False Positive Rate", x=0.43)
sup_ylabel = p1.fig.supylabel("True Positive Rate")
p1.set(xlabel=None, ylabel=None, ylim=(0, 1), xlim=(0, 1))

for ax in p1.axes.flat:
    # Turn on minor grids
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(True, which="minor", axis="both", linewidth=0.25)
    # Remove Margins
    ax.margins(0)

# Figure Size
p1.fig.set_figwidth(12.0)
p1.fig.set_figheight(10.0)

# Tight Layout
p1.tight_layout()
# %%
# Save Figure
p1.fig.savefig(
    "figure_ROC_curves.png",
    dpi=300,
    bbox_extra_artists=(p1.legend, sup_title, sup_xlabel, sup_ylabel),
    bbox_inches="tight",
)

# %% [markdown]
# ## Precision-Recall Curve Figure
# %%
# Facet Grid (like a subplot) of Precision-Recall Curve for all ASA-PS for all 6 models
sns.set_style("whitegrid")
sns.plotting_context("poster")
sns.color_palette("colorblind")
sns.despine(bottom=True, left=True)
p2 = sns.relplot(
    kind="line",
    data=pr_df,
    x="Recall",
    y="Precision",
    hue="Model",
    style="Model",
    errorbar=None,
    col="ASA",
    col_wrap=2,
    dashes={
        "Random Forest": (4, 4),
        "Support Vector Machine": (3, 3),
        "fastText": (2, 2),
        "BioClinicalBERT": (1, 1),
        "Age & Meds": (8, 8),
        "Random Classifier": (8, 8),
    },
    facet_kws={
        "sharex": True,
        "sharey": True,
        "legend_out": True,
        "despine": False,
    },
)

# Reformat Subplot Titles to be "ASA x"
p2.set_titles(template="{col_var} {col_name}")

# Overall Figure Title
sup_title = p2.fig.suptitle("Precision-Recall Curves for Note512 Task", fontsize=16)
# Share x-label & y-label
sup_xlabel = p2.fig.supxlabel("Recall", x=0.43)
sup_ylabel = p2.fig.supylabel("Precision")
p2.set(xlabel=None, ylabel=None, ylim=(0, 1), xlim=(0, 1))

# Turn on minor grids
for ax in p2.axes.flat:
    # Turn on minor grids
    ax.get_xaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.grid(True, which="minor", axis="both", linewidth=0.25)
    # Remove Margins
    ax.margins(0)

# Figure Size
p2.fig.set_figwidth(12.0)
p2.fig.set_figheight(10.0)

# Tight Layout
p2.tight_layout()
# %%
# Save Figure
p2.fig.savefig(
    "figure_PR_curves.png",
    dpi=300,
    bbox_extra_artists=(p2.legend, sup_title, sup_xlabel, sup_ylabel),
    bbox_inches="tight",
)
# %%
