# %% [markdown]
# ## Contingency Tables Figure
# Generate contingency table figure.  Run `evaluate_all_models.py` script first.
# %%
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pytorch_lightning import seed_everything
from results_dataclasses import MetricResults, ModelPredictions
from tqdm.auto import tqdm

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
class_labels = note512_predictions.asa_class_names

# Load Metrics For All Tasks & Models
metric_results = {}
for task_input, task_predictions in (pbar1 := tqdm(model_predictions.items())):
    metric_results_path = current_dir / f"metric_results_{task_input}.pkl"
    with open(metric_results_path, "rb") as file:
        metric_results[task_input] = pickle.load(file)

note512_metric_results = metric_results["note512"]
# %% [markdown]
# ## 4x4 Contingency Tables (Confusion Matrices)
# %%
# Generate Confusion Matrix Plot comparing all Models on Note512 Task


def annotate_col_normalized_percent(s1: pd.Series, s2: pd.Series) -> pd.Series:
    outputs = []
    for item1, item2 in zip(s1, s2):
        text = f"{item1:.0f}\n({item2:.1%})"
        outputs += [text]
    return pd.Series(data=outputs, index=s1.index)


cmap = "Blues"

fig, axes = plt.subplots(
    nrows=3, ncols=2, sharex=True, sharey=True, figsize=(8, 10), dpi=300
)
# Random Forest
raw_cm = pd.DataFrame(
    note512_metric_results.rf["ConfusionMatrix/Raw"],
    columns=class_labels,
    index=class_labels,
)
norm_cm = pd.DataFrame(
    note512_metric_results.rf["ConfusionMatrix/NormalizeOverPredictions"],
    columns=class_labels,
    index=class_labels,
)
annot_cm = raw_cm.combine(norm_cm, func=annotate_col_normalized_percent)
p1 = sns.heatmap(
    ax=axes[0, 0],
    data=raw_cm,
    annot=annot_cm,
    fmt="",
    cmap=cmap,
    cbar=False,
)
p1.set(title="Random Forest")
p1.tick_params(left=False, bottom=False)
# Support Vector Machine
raw_cm = pd.DataFrame(
    note512_metric_results.svm["ConfusionMatrix/Raw"],
    columns=class_labels,
    index=class_labels,
)
norm_cm = pd.DataFrame(
    note512_metric_results.svm["ConfusionMatrix/NormalizeOverPredictions"],
    columns=class_labels,
    index=class_labels,
)
annot_cm = raw_cm.combine(norm_cm, func=annotate_col_normalized_percent)
p2 = sns.heatmap(
    ax=axes[0, 1],
    data=raw_cm,
    annot=annot_cm,
    fmt="",
    cmap=cmap,
    cbar=False,
)
p2.set(title="Support Vector Machine")
p2.tick_params(left=False, bottom=False)
# fastText
raw_cm = pd.DataFrame(
    note512_metric_results.fasttext["ConfusionMatrix/Raw"],
    columns=class_labels,
    index=class_labels,
)
norm_cm = pd.DataFrame(
    note512_metric_results.fasttext["ConfusionMatrix/NormalizeOverPredictions"],
    columns=class_labels,
    index=class_labels,
)
annot_cm = raw_cm.combine(norm_cm, func=annotate_col_normalized_percent)
p3 = sns.heatmap(
    ax=axes[1, 0],
    data=raw_cm,
    annot=annot_cm,
    fmt="",
    cmap=cmap,
    cbar=False,
)
p3.set(title="fastText")
p3.tick_params(left=False, bottom=False)
# BioClinicalBERT
raw_cm = pd.DataFrame(
    note512_metric_results.bioclinicalbert["ConfusionMatrix/Raw"],
    columns=class_labels,
    index=class_labels,
)
norm_cm = pd.DataFrame(
    note512_metric_results.bioclinicalbert["ConfusionMatrix/NormalizeOverPredictions"],
    columns=class_labels,
    index=class_labels,
)
annot_cm = raw_cm.combine(norm_cm, func=annotate_col_normalized_percent)
p4 = sns.heatmap(
    ax=axes[1, 1],
    data=raw_cm,
    annot=annot_cm,
    fmt="",
    cmap=cmap,
    cbar=False,
)
p4.set(title="BioClinicalBERT")
p4.tick_params(left=False, bottom=False)
# Age & Meds Classifier
raw_cm = pd.DataFrame(
    note512_metric_results.age_meds["ConfusionMatrix/Raw"],
    columns=class_labels,
    index=class_labels,
)
norm_cm = pd.DataFrame(
    note512_metric_results.age_meds["ConfusionMatrix/NormalizeOverPredictions"],
    columns=class_labels,
    index=class_labels,
)
annot_cm = raw_cm.combine(norm_cm, func=annotate_col_normalized_percent)
p5 = sns.heatmap(
    ax=axes[2, 0],
    data=raw_cm,
    annot=annot_cm,
    fmt="",
    cmap=cmap,
    cbar=False,
)
p5.set(title="Age & Meds Classifier")
p5.tick_params(left=False, bottom=False)

raw_cm = pd.DataFrame(
    note512_metric_results.random_classifier["ConfusionMatrix/Raw"],
    columns=class_labels,
    index=class_labels,
)
norm_cm = pd.DataFrame(
    note512_metric_results.random_classifier[
        "ConfusionMatrix/NormalizeOverPredictions"
    ],
    columns=class_labels,
    index=class_labels,
)
annot_cm = raw_cm.combine(norm_cm, func=annotate_col_normalized_percent)
p6 = sns.heatmap(
    ax=axes[2, 1],
    data=raw_cm,
    annot=annot_cm,
    fmt="",
    cmap=cmap,
    cbar=False,
)
p6.set(title="Random Classifier")
p6.tick_params(left=False, bottom=False)

# # Overall Figure Title
sup_title = fig.suptitle(
    "Anesthesiologist Assigned ASA-PS vs. Model Predictions on Note512 Task",
    fontsize=16,
)
sup_xlabel = fig.supxlabel("Model Predicted ASA-PS", x=0.45)
sup_ylabel = fig.supylabel("Anesthesiologist Assigned ASA-PS")

# Tight Layout
fig.tight_layout(pad=1.2, h_pad=3, w_pad=3)

# Create Colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar_ax.set_title("Number of \nCases", pad=20)
m = plt.cm.ScalarMappable(cmap=cmap)
m.set_clim(0, 5000)
cbar = fig.colorbar(m, cax=cbar_ax, orientation="vertical")
cbar.ax.tick_params(size=0)

fig.show()

# %%
fig.savefig(
    "figure_confusion_matrix.png",
    dpi=300,
    bbox_extra_artists=(sup_title, sup_xlabel, sup_ylabel),
    bbox_inches="tight",
)
# %%
