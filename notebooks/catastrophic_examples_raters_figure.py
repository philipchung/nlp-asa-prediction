# %%
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torchmetrics.functional import mean_absolute_error

pd.options.display.float_format = "{:,.3f}".format

# %% [markdown]
# ## Load Human-Annotated Dataset
# Legend (ASA-PS = Numeric Scale = Dataset Mapped Value):
# * ASA I = 1 = 0
# * ASA II = 2 = 1
# * ASA III = 3 = 2
# * ASA IV = 4 = 3
# * ASA V = 4 = 3
# %%
df = pd.read_csv("error_analysis_annotated_dataset.tsv", sep="\t")
# Drop NaN rows
df = df.dropna(how="all", axis=0)
# Normalize Dataframe so all columns are numeric
df = (
    df.loc[
        :,
        [
            "original_index",
            "target numeric",
            "prediction numeric",
            "rater0",
            "rater1",
            "rater2",
        ],
    ]
    .rename(columns={"target numeric": "label", "prediction numeric": "model_preds"})
    .astype(int)
    .set_index("original_index")
)
# Force ASA 5 into ASA 4
df = df.applymap(lambda x: x if x < 4 else 4)
# Get Mean and Standard Deviation for Raters
df["rater_mean"] = df.loc[:, ["rater0", "rater1", "rater2"]].mean(axis=1)
df["rater_std"] = df.loc[:, ["rater0", "rater1", "rater2"]].std(axis=1)
df

# %%
# All examples where Actual=I, Predicted=IV-V
target1_pred4_examples = df.loc[df.label == 1, :]
# All examples where Actual=IV-V, Predicted=I
target4_pred1_examples = df.loc[df.label == 4, :]

# %%
## Visualize Rater's ASA-PS for All examples
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), dpi=300)
p1, p2 = axes

# Subplot for all examples where Anesthesiologist=I, Model Prediction=IV-V
sns.swarmplot(
    ax=p1, data=target1_pred4_examples.loc[:, ["rater0", "rater1", "rater2"]].T
)
# Annotate & Highlight Model Prediction ASA-PS IV-V
red_color = matplotlib.colors.TABLEAU_COLORS["tab:red"]
p1.text(9, 4.4, "Model Prediction", color=red_color)
p1.axhspan(ymin=3.75, ymax=4.25, alpha=0.15, color=red_color)
# Annotate & Highlight Target ASA-PS I
green_color = matplotlib.colors.TABLEAU_COLORS["tab:green"]
p1.text(8.5, 0.5, "Original Anesthesiologist", color=green_color)
p1.axhspan(
    ymin=0.75,
    ymax=1.25,
    alpha=0.15,
    color=green_color,
)
# Format Plot
p1.tick_params(axis="x", labelrotation=45)
p1.set(
    title="Original Anesthesiologist: I \nModel Prediction: IV-V",
    xlabel="Case Number from Dataset",
    ylabel="ASA-PS Assigned by Rater",
    ylim=[0, 5],
    yticks=[1, 2, 3, 4],
    yticklabels=["I", "II", "III", "IV-V"],
)
p1.grid(True, alpha=0.5)

# Subplot for all examples where Anesthesiologist=IV-V, Model Prediction=I
sns.swarmplot(
    ax=p2, data=target4_pred1_examples.loc[:, ["rater0", "rater1", "rater2"]].T
)
# Annotate & Highlight Target ASA-PS I
green_color = matplotlib.colors.TABLEAU_COLORS["tab:green"]
p2.text(7.5, 4.4, "Original Anesthesiologist", color=green_color)
p2.axhspan(
    ymin=3.75,
    ymax=4.25,
    alpha=0.15,
    color=green_color,
)
# Annotate & Highlight Model Prediction ASA-PS IV-V
red_color = matplotlib.colors.TABLEAU_COLORS["tab:red"]
p2.text(8, 0.5, "Model Prediction", color=red_color)
p2.axhspan(
    ymin=0.75,
    ymax=1.25,
    alpha=0.15,
    color=red_color,
)
# Format Plot
p2.tick_params(axis="x", labelrotation=45)
p2.set(
    title="Original Anesthesiologist: IV-V \nModel Prediction: I",
    xlabel="Case Number from Dataset",
    ylabel="ASA-PS Assigned by Rater",
    ylim=[0, 5],
    yticks=[1, 2, 3, 4],
    yticklabels=["I", "II", "III", "IV-V"],
)
p2.grid(True, alpha=0.5)

# Overall Figure Formatting
sup_title = plt.suptitle(
    "Rater Assignments of ASA-PS for Catastrophic Error Examples", fontsize=16
)
plt.tight_layout()

# %%
fig.savefig(
    "figure_raters_agreement.png",
    dpi=300,
    bbox_extra_artists=(sup_title,),
    bbox_inches="tight",
)

# %%
# Mean absolute error between mean of Rater ASA-PS vs. Label
preds = torch.from_numpy(df.rater_mean.astype(float).to_numpy())
target = torch.from_numpy(df.label.astype(float).to_numpy())
rater_mean_pearson_corr = mean_absolute_error(preds=preds, target=target)
print("Rater Mean vs. Label MAE:", rater_mean_pearson_corr)
# %%
# Mean absolute error between Model Prediction vs. Label
preds = torch.from_numpy(df.model_preds.astype(float).to_numpy())
target = torch.from_numpy(df.label.astype(float).to_numpy())
model_preds_pearson_corr = mean_absolute_error(preds=preds, target=target)
print("Model Prediction vs. Label MAE:", model_preds_pearson_corr)
# %%
# Mean absolute error between Rater ASA-PS vs. Model Prediction
preds = torch.from_numpy(df.rater_mean.astype(float).to_numpy())
target = torch.from_numpy(df.model_preds.astype(float).to_numpy())
model_preds_pearson_corr = mean_absolute_error(preds=preds, target=target)
print("Rater vs. Model Prediction MAE:", model_preds_pearson_corr)
# %%
