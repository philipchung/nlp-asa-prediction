#%% [markdown]
# ## ROC & PR Curve Figures
# Generate ROC and PR figures.  Run `evaluate_all_models.py` script first.
#%%
from __future__ import annotations

import pickle

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torchmetrics.functional.classification import (
    multiclass_precision_recall_curve,
    multiclass_roc,
)

#%% [markdown]
# ## Load Results of Model Evaluation
#%%
# Load Evaluation Results
with open("model_predictions.pkl", "rb") as file:
    results = pickle.load(file)

class_labels = results["asa_class_names"]
#%% [markdown]
# ## ROC Curves
#%%
num_thresholds = None


def compute_roc(
    preds: torch.tensor | np.array,
    target: torch.tensor | np.array,
    num_thresholds: int = 100,
):
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    roc_dict = dict(
        zip(
            ["fpr", "tpr", "thresholds"],
            multiclass_roc(
                preds=preds,
                target=target,
                num_classes=4,
                thresholds=num_thresholds,
            ),
        )
    )
    return roc_dict


def reformat_roc(
    roc_dict: dict[str, torch.Tensor], class_names: tuple | list
) -> pd.DataFrame:
    roc_data = []
    # Extract ROC by Class
    for idx, class_name in enumerate(class_names):
        class_df = pd.DataFrame.from_dict(
            {"FPR": roc_dict["fpr"][idx], "TPR": roc_dict["tpr"][idx]}
        )
        class_df["ASA"] = class_name
        roc_data += [class_df]
    # Combine ROC from all Classes into single long dataframe
    roc_df = pd.concat(roc_data, axis=0)
    return roc_df


roc_data = []

# Random Forest ROC Curves
rf_roc = compute_roc(
    preds=torch.from_numpy(results["rf_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
rf_roc = reformat_roc(roc_dict=rf_roc, class_names=class_labels)
rf_roc["Model"] = "Random Forest"
roc_data += [rf_roc]

# Support Vector Machine ROC Curves
svm_roc = compute_roc(
    preds=torch.from_numpy(results["svm_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
svm_roc = reformat_roc(roc_dict=svm_roc, class_names=class_labels)
svm_roc["Model"] = "Support Vector Machine"
roc_data += [svm_roc]

# fastText ROC Curves
fasttext_roc = compute_roc(
    preds=torch.from_numpy(results["fasttext_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
fasttext_roc = reformat_roc(roc_dict=fasttext_roc, class_names=class_labels)
fasttext_roc["Model"] = "fastText"
roc_data += [fasttext_roc]

# BioClinicalBERT ROC Curves
bioclinicalbert_roc = compute_roc(
    preds=torch.from_numpy(results["bioclinicalbert_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
bioclinicalbert_roc = reformat_roc(
    roc_dict=bioclinicalbert_roc, class_names=class_labels
)
bioclinicalbert_roc["Model"] = "BioClinicalBERT"
roc_data += [bioclinicalbert_roc]

# Age Classifier ROC Curves
age_classifier_roc = compute_roc(
    preds=torch.from_numpy(results["age_classifier_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
age_classifier_roc = reformat_roc(roc_dict=age_classifier_roc, class_names=class_labels)
age_classifier_roc["Model"] = "Age Classifier"
roc_data += [age_classifier_roc]

# Random Classifier ROC Curves
random_classifier_roc = compute_roc(
    preds=torch.from_numpy(results["random_classifier_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
random_classifier_roc = reformat_roc(
    roc_dict=random_classifier_roc, class_names=class_labels
)
random_classifier_roc["Model"] = "Random Classifier"
roc_data += [random_classifier_roc]

# Combine All ROC Data into Single DataFrame
roc_data = pd.concat(roc_data, axis=0)
#%%
# Facet Grid (like a subplot) of ROC Curve for all ASA-PS for all 6 models
sns.set_style("whitegrid")
sns.plotting_context("poster")
sns.color_palette("colorblind")
p1 = sns.relplot(
    kind="line",
    data=roc_data,
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
        "Age Classifier": (8, 8),
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
#%%
# Save Figure
p1.fig.savefig(
    "figure_ROC_curves.png",
    dpi=300,
    bbox_extra_artists=(p1.legend, sup_title, sup_xlabel, sup_ylabel),
    bbox_inches="tight",
)

#%% [markdown]
# ## Precision-Recall Curves
#%%
num_thresholds = None


def compute_pr(
    preds: torch.tensor | np.array,
    target: torch.tensor | np.array,
    num_thresholds: int = 100,
):
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    pr_dict = dict(
        zip(
            ["precision", "recall", "thresholds"],
            multiclass_precision_recall_curve(
                preds=preds,
                target=target,
                num_classes=4,
                thresholds=num_thresholds,
            ),
        )
    )
    return pr_dict


def reformat_pr(
    pr_dict: dict[str, torch.Tensor], class_names: tuple | list
) -> pd.DataFrame:
    pr_data = []
    # Extract ROC by Class
    for idx, class_name in enumerate(class_names):
        class_df = pd.DataFrame.from_dict(
            {"Precision": pr_dict["precision"][idx], "Recall": pr_dict["recall"][idx]}
        )
        class_df["ASA"] = class_name
        pr_data += [class_df]
    # Combine ROC from all Classes into single long dataframe
    pr_df = pd.concat(pr_data, axis=0)
    return pr_df


pr_data = []

# Random Forest Precision-Recall Curves
rf_pr = compute_pr(
    preds=torch.from_numpy(results["rf_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
rf_pr = reformat_pr(pr_dict=rf_pr, class_names=class_labels)
rf_pr["Model"] = "Random Forest"
pr_data += [rf_pr]

# Support Vector Machine Precision-Recall Curves
svm_pr = compute_pr(
    preds=torch.from_numpy(results["svm_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
svm_pr = reformat_pr(pr_dict=svm_pr, class_names=class_labels)
svm_pr["Model"] = "Support Vector Machine"
pr_data += [svm_pr]

# fastText Precision-Recall Curves
fasttext_pr = compute_pr(
    preds=torch.from_numpy(results["fasttext_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
fasttext_pr = reformat_pr(pr_dict=fasttext_pr, class_names=class_labels)
fasttext_pr["Model"] = "fastText"
pr_data += [fasttext_pr]

# BioClinicalBERT Precision-Recall Curves
bioclinicalbert_pr = compute_pr(
    preds=torch.from_numpy(results["bioclinicalbert_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
bioclinicalbert_pr = reformat_pr(pr_dict=bioclinicalbert_pr, class_names=class_labels)
bioclinicalbert_pr["Model"] = "BioClinicalBERT"
pr_data += [bioclinicalbert_pr]

# Age Classifier Precision-Recall Curves
age_classifier_pr = compute_pr(
    preds=torch.from_numpy(results["age_classifier_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
age_classifier_pr = reformat_pr(pr_dict=age_classifier_pr, class_names=class_labels)
age_classifier_pr["Model"] = "Age Classifier"
pr_data += [age_classifier_pr]

# Random Classifier Precision-Recall Curves
random_classifier_pr = compute_pr(
    preds=torch.from_numpy(results["random_classifier_pred_proba"]),
    target=torch.from_numpy(results["labels"]),
    num_thresholds=num_thresholds,
)
random_classifier_pr = reformat_pr(
    pr_dict=random_classifier_pr, class_names=class_labels
)
random_classifier_pr["Model"] = "Random Classifier"


def force_precision_to_class_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Corrects random classifier PR curve to be a flat horizontal line corresponding
    to the class's frequency.
    In torchmetrics, when computing PR curves, initial Precision always starts at 0.0 or 1.0.
    But will result in a non-truthful curve.
    """
    precision_class_frequency = df.loc[df.Recall == 1.0].Precision.iloc[0]
    df.Precision = precision_class_frequency
    return df


random_classifier_pr = random_classifier_pr.groupby("ASA", group_keys=False).apply(
    force_precision_to_class_frequency
)


# random_classifier_pr = random_classifier_pr.loc[random_classifier_pr.Recall == 1.0]
pr_data += [random_classifier_pr]

# Combine All Precision-Recall Data into Single DataFrame
pr_data = pd.concat(pr_data, axis=0)

# %%
# Facet Grid (like a subplot) of Precision-Recall Curve for all ASA-PS for all 6 models
sns.set_style("whitegrid")
sns.plotting_context("poster")
sns.color_palette("colorblind")
sns.despine(bottom=True, left=True)
p2 = sns.relplot(
    kind="line",
    data=pr_data,
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
        "Age Classifier": (8, 8),
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
#%%
# Save Figure
p2.fig.savefig(
    "figure_PR_curves.png",
    dpi=300,
    bbox_extra_artists=(p2.legend, sup_title, sup_xlabel, sup_ylabel),
    bbox_inches="tight",
)
#%%
