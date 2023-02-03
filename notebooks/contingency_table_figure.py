#%% [markdown]
# ## Contingency Tables Figure
# Generate contingency table figure.  Run `evaluate_all_models.py` script first.
#%%
from __future__ import annotations

import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.metrics.functional import confusion_matrix_metrics

#%% [markdown]
# ## Load Results of Model Evaluation
#%%
# Load Evaluation Results
with open("model_predictions.pkl", "rb") as file:
    results = pickle.load(file)

class_labels = results["asa_class_names"]
#%% [markdown]
# ## 4x4 Contingency Tables (Confusion Matrices)

# %%
# Generate Confusion Matrices for all Models
conf_mat = {
    "rf": confusion_matrix_metrics(preds=results["rf_preds"], target=results["labels"]),
    "svm": confusion_matrix_metrics(
        preds=results["svm_preds"], target=results["labels"]
    ),
    "fasttext": confusion_matrix_metrics(
        preds=results["fasttext_preds"], target=results["labels"]
    ),
    "bioclinicalbert": confusion_matrix_metrics(
        preds=results["bioclinicalbert_preds"], target=results["labels"]
    ),
    "age_classifier": confusion_matrix_metrics(
        preds=results["age_classifier_preds"], target=results["labels"]
    ),
    "random_classifier": confusion_matrix_metrics(
        preds=results["random_classifier_preds"], target=results["labels"]
    ),
}
conf_mat_df = pd.DataFrame.from_dict(conf_mat, orient="index")
conf_mat_df
#%%
# Generate Confusion Matrix Plot comparing all Models on Note512 Task
conf_mat_type = "Raw"

cmap = "Blues"

fig, axes = plt.subplots(
    nrows=3, ncols=2, sharex=True, sharey=True, figsize=(8, 10), dpi=300
)

cm = conf_mat_df.loc["rf", f"ConfusionMatrix/{conf_mat_type}"]
df_cm = pd.DataFrame(cm, columns=class_labels, index=class_labels)
p1 = sns.heatmap(
    ax=axes[0, 0],
    data=df_cm,
    cmap=cmap,
    cbar=False,
    annot=True,
    fmt="g",
)
p1.set(title="Random Forest")
p1.tick_params(left=False, bottom=False)

cm = conf_mat_df.loc["svm", f"ConfusionMatrix/{conf_mat_type}"]
df_cm = pd.DataFrame(cm, columns=class_labels, index=class_labels)
p2 = sns.heatmap(
    ax=axes[0, 1],
    data=df_cm,
    cmap=cmap,
    cbar=False,
    annot=True,
    fmt="g",
)
p2.set(title="Support Vector Machine")
p2.tick_params(left=False, bottom=False)

cm = conf_mat_df.loc["fasttext", f"ConfusionMatrix/{conf_mat_type}"]
df_cm = pd.DataFrame(cm, columns=class_labels, index=class_labels)
p3 = sns.heatmap(
    ax=axes[1, 0],
    data=df_cm,
    cmap=cmap,
    cbar=False,
    annot=True,
    fmt="g",
)
p3.set(title="fastText")
p3.tick_params(left=False, bottom=False)

cm = conf_mat_df.loc["bioclinicalbert", f"ConfusionMatrix/{conf_mat_type}"]
df_cm = pd.DataFrame(cm, columns=class_labels, index=class_labels)
p4 = sns.heatmap(
    ax=axes[1, 1],
    data=df_cm,
    cmap=cmap,
    cbar=False,
    annot=True,
    fmt="g",
)
p4.set(title="BioClinicalBERT")
p4.tick_params(left=False, bottom=False)

cm = conf_mat_df.loc["age_classifier", f"ConfusionMatrix/{conf_mat_type}"]
df_cm = pd.DataFrame(cm, columns=class_labels, index=class_labels)
p5 = sns.heatmap(
    ax=axes[2, 0],
    data=df_cm,
    cmap=cmap,
    cbar=False,
    annot=True,
    fmt="g",
)
p5.set(title="Age Classifier")
p5.tick_params(left=False, bottom=False)

cm = conf_mat_df.loc["random_classifier", f"ConfusionMatrix/{conf_mat_type}"]
df_cm = pd.DataFrame(cm, columns=class_labels, index=class_labels)
p6 = sns.heatmap(
    ax=axes[2, 1],
    data=df_cm,
    cmap=cmap,
    cbar=False,
    annot=True,
    fmt="g",
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
#%%
fig.savefig(
    "figure_confusion_matrix.png",
    dpi=300,
    bbox_extra_artists=(sup_title, sup_xlabel, sup_ylabel),
    bbox_inches="tight",
)
#%%
