#%% [markdown]
# ## Dataset Statistics for Table 1: Baseline Statistics
# Generate statistics for Patient Characteristics, Case Characteristics, Note Characteristics
#%%
import logging
import re
from functools import partial
from pathlib import Path
from typing import Any, Union, Iterable

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from src.dataset.data import Data
from src.modeling.datamodule import DataModule

dataset_split_dtype = CategoricalDtype(["train", "validation", "test"], ordered=True)

project_root_dir = Path(__file__).parent.parent

logger = logging.getLogger("dataset_logger")

# Load final datasets used for training
datamodule = DataModule(
    project_dir=project_root_dir,
    dataloader_batch_size=32,
    seq_max_length=512,
    dataset_path=Path(__file__).parent / "data",
)
datamodule.prepare_data()
#%%
train_ds = datamodule.train
validation_ds = datamodule.validation
test_ds = datamodule.test
train_df = datamodule.train_dataframe()
train_df["split"] = "train"
train_df.split = train_df.split.astype(dataset_split_dtype)
validation_df = datamodule.val_dataframe()
validation_df["split"] = "validation"
validation_df.split = validation_df.split.astype(dataset_split_dtype)
test_df = datamodule.test_dataframe()
test_df["split"] = "test"
test_df.split = test_df.split.astype(dataset_split_dtype)

all_df = pd.concat([train_df, validation_df, test_df], axis=0)
# %%
# We will need to join demographic data from raw notes & cases tables
data = Data(
    project_dir=project_root_dir,
    datastore_blob_path=datamodule.data_blob_prefix,
    dataset_name=datamodule.dataset_name,
    seed=datamodule.seed,
)
cases = data.get_cases()
notes = data.get_notes()
#%%
# Num patients, cases, notes in dataset
num_patients = len(all_df.PersonID.unique())
print(f"There are {num_patients} patients in the dataset.")
num_cases = len(all_df.ProcedureID.unique())
print(f"There are {num_cases} cases in the dataset.")
num_notes = len(all_df.PreAnesthesiaNoteID.unique())
print(f"There are {num_notes} notes in the dataset.")

#%%
# Utility Functions
def string_from_template(template: str = "", **kwargs):
    """Unpacks `kwargs` by string key and inserts them into `template`.

    Example:
    kwargs = {"value1": 10, "value2": "pizzas"}
    template = "I want to eat {value1} {value2} today."

    Function Signature:
    string_from_template(value1=10, value2="pizzas", template="I want to eat {value1} {value2} today.")

    Result: "I want to eat 10 pizzas today."
    """
    return template.format(**kwargs)


def elementwise_string_from_template(template: str = "", **kwargs):
    """Creates string elementwise across dataframes specified in `kwargs`.

    Example:
    df1 = pd.DataFrame({"a": [1, 11], "b": [2, 22], "c": [3, 33]})
    df2 = pd.DataFrame({"a": [4, 44], "b": [5, 55], "c": [6, 66]})
    df3 = pd.DataFrame({"a": [7, 77], "b": [8, 88], "c": [9, 99]})

    Function Signature:
    elementwise_string_from_template(template="{a}-{b}-{c}", a=df1, b=df2, c=df3)

    Result: pandas Dataframe with templated string elements.
        a	        b	        c
    0	1-4-7	    2-5-8	    3-6-9
    1	11-44-77	22-55-88	33-66-99
    """
    # Vectorized string template method for elementwise operations
    vec_fn = np.vectorize(string_from_template)
    result = vec_fn(template=template, **kwargs)
    # Result = array of strings for pd.Series, or a 2D array of strings for pd.DataFrame
    # This is a numpy array.  Dataframe index & columns, or Series index info is lost.

    # We now need to reconstruct indices/columns
    # Get first object from kwargs dict
    pd_obj = next(iter(kwargs.values()))
    # If pd.DataFrame
    if isinstance(pd_obj, pd.DataFrame):
        result = pd.DataFrame(data=result, index=pd_obj.index, columns=pd_obj.columns)
    # If pd.Series
    elif isinstance(pd_obj, pd.Series):
        result = pd.Series(data=result, index=pd_obj.index)
    else:
        raise ValueError("Values on `kwargs` must all be pd.Series or pd.DataFrame.")
    return result


def count_to_percent(df: Union[pd.Series, pd.DataFrame], axis: int = 0):
    "String format counts with percentage.  `axis` determines dimension for which % is calculated."
    total_counts = df.sum(axis=axis)
    if axis == 0 or isinstance(df, pd.Series):
        df_percent = df / total_counts * 100
    elif axis == 1:
        df_percent = (df.T / total_counts * 100).T
    else:
        raise ValueError("Argument `axis` must be 0 or 1.")
    return df_percent


def agg_sum_past_threshold(df: pd.DataFrame, n=5) -> pd.DataFrame:
    "For rows that have column values >=n, will aggregate (sum) values and create new row `n+`."
    mask = df.index >= n
    rows_to_keep = df.loc[~mask]
    rows_to_combine = df.loc[mask]
    combined_rows = rows_to_combine.sum().rename(f">={n}").to_frame().T
    result = pd.concat([rows_to_keep, combined_rows])
    result.columns = result.columns.rename("")
    return result


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)

    percentile_.__name__ = "percentile_%s" % n
    return percentile_


#%% [markdown]
# ## Patient Characteristics
# %%
# Patient count in train, validation, test split
patient_ct = all_df.groupby("split")["PersonID"].apply(
    lambda series: len(series.unique())
)
patient_pct = count_to_percent(patient_ct).round(decimals=1)
patient_ct_table = (
    elementwise_string_from_template(template="{a} ({b}%)", a=patient_ct, b=patient_pct)
    .rename("Patient Count")
    .to_frame()
    .T
)
patient_ct_table

#%%
# Number of Surgeries per Patient
case_ct_per_patient = (
    all_df.groupby("PersonID")[["ProcedureID", "split"]]
    .agg({"ProcedureID": "count", "split": lambda x: list(set(x))[0]})
    .rename(columns={"ProcedureID": "num_cases"})
)

# How may patients had 1 surgery, how many had 2 surgeries... etc.
patient_ct_for_case_ct = (
    case_ct_per_patient.value_counts().unstack("split").fillna(0).astype(int)
)

# Collapse patients with >5 surgeries into a single bin
patient_ct_for_case_ct = agg_sum_past_threshold(patient_ct_for_case_ct, n=5)
patient_ct_for_case_ct.index = patient_ct_for_case_ct.index.rename(
    "Number of Surgeries per Patient"
)
patient_pct_for_case_ct = count_to_percent(patient_ct_for_case_ct, axis=0).round(
    decimals=2
)

# Combine Count and Percent into same table
patient_ct_for_case_ct_table = elementwise_string_from_template(
    template="{a} ({b}%)", a=patient_ct_for_case_ct, b=patient_pct_for_case_ct
)
patient_ct_for_case_ct_table

#%%
# Age
demographics_df = all_df.set_index("ProcedureID")[["split"]].join(
    cases.loc[:, ["Age", "Gender"]]
)

age_stats = (
    demographics_df.groupby("split")["Age"]
    .apply(lambda series: {"mean": series.mean(), "std": series.std()})
    .round(decimals=2)
    .unstack("split")
)
age_stats_table = (
    elementwise_string_from_template(
        template="{a} ({b})", a=age_stats.loc["mean", :], b=age_stats.loc["std", :]
    )
    .rename("Age")
    .to_frame()
    .T
)
age_stats_table
#%%
# Gender
gender_ct = (
    demographics_df.groupby("split")[["Gender"]]
    .value_counts()
    .unstack("Gender")
    .fillna(0)
    .astype(int)
    .T
)
gender_pct = count_to_percent(gender_ct, axis=1).round(decimals=2)
gender_ct_table = elementwise_string_from_template(
    template="{a} ({b}%)", a=gender_ct, b=gender_pct
)
gender_ct_table
#%% [markdown]
# ## Case Characteristics
#%%
# Case Count
cases_df = all_df.set_index("ProcedureID")[["split", "asa_class", "emergency"]].join(
    cases.loc[:, ["AnesType"]]
)
case_ct = cases_df.split.value_counts()
case_pct = count_to_percent(case_ct).round(decimals=2)
case_ct_table = (
    elementwise_string_from_template(template="{a} ({b}%)", a=case_ct, b=case_pct)
    .rename("Case Count")
    .to_frame()
    .T
)
case_ct_table

#%%
# Anesthesia Type: General, MAC, Regional
anes_type_ct = cases_df.groupby("split")[["AnesType"]].value_counts().unstack("split")
anes_type_pct = count_to_percent(anes_type_ct).round(decimals=2)
anes_type_table = elementwise_string_from_template(
    template="{a} ({b}%)", a=anes_type_ct, b=anes_type_pct
)
anes_type_table

#%%
# ASA Physical Status Classification
asa_ct = cases_df.groupby("split")[["asa_class"]].value_counts().unstack("split")
# asa_ct = cases_df.groupby("split")[["asa_class", "emergency"]].value_counts().unstack("split")
asa_pct = count_to_percent(asa_ct).round(decimals=2)
asa_ct_table = elementwise_string_from_template(
    template="{a} ({b}%)", a=asa_ct, b=asa_pct
)
asa_ct_table
#%%
# Time Since Last Pre-Anesthesia Note
case_note_df = all_df.set_index("ProcedureID")[["PreAnesthesiaNoteID", "split"]].join(
    cases[["AnesStart"]]
)
case_note_df = pd.merge(
    left=case_note_df,
    right=notes[["LastUpdateDateTime"]],
    left_on="PreAnesthesiaNoteID",
    right_index=True,
)
case_note_df["TimeDelta"] = case_note_df.AnesStart - case_note_df.LastUpdateDateTime
case_note_timedelta = (
    case_note_df.groupby("split")["TimeDelta"]
    .agg([np.median, percentile(25), percentile(75)])
    .apply(lambda timedelta_obj: timedelta_obj.round("1s"))
    .unstack("split")
)
case_note_timedelta_table = (
    elementwise_string_from_template(
        template="{a} ({b}, {c})",
        a=case_note_timedelta.loc["median", :].astype(str),
        b=case_note_timedelta.loc["percentile_25", :].astype(str),
        c=case_note_timedelta.loc["percentile_75", :].astype(str),
    )
    .rename("Time Between Pre-Anesthesia Note and Surgery")
    .to_frame()
    .T
)
case_note_timedelta_table
#%% [markdown]
# ## Note Characteristcs
#%%
# Note Count
notes_ct_df = (
    all_df[["PreAnesthesiaNoteID", "split"]]
    .groupby("split")["PreAnesthesiaNoteID"]
    .apply(lambda series: len(series.unique()))
)
notes_pct_df = count_to_percent(notes_ct_df).round(decimals=2)
notes_ct_table = (
    elementwise_string_from_template(
        template="{a} ({b}%)", a=notes_ct_df, b=notes_pct_df
    )
    .rename("Notes Count")
    .to_frame()
    .T
)
notes_ct_table
#%%
# Note, word-level length (whitespace delimited)
notes_df = (
    all_df[["PreAnesthesiaNoteID", "split", "ProcedureID"]]
    .drop_duplicates(subset="PreAnesthesiaNoteID")
    .set_index("PreAnesthesiaNoteID")
    .join(notes[["NoteText", "ROS", "HPI", "PMSH", "Medications"]])
)
notes_df = pd.merge(
    left=notes_df,
    right=cases[["ProcedureDescription", "DiagnosisDescription"]],
    left_on="ProcedureID",
    right_index=True,
).drop(columns="ProcedureID")
notes_split_df = notes_df[["split"]]
notes_text_df = notes_df[
    [
        "NoteText",
        "ProcedureDescription",
        "DiagnosisDescription",
        "HPI",
        "PMSH",
        "ROS",
        "Medications",
    ]
]


def remove_whitespace(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence, flags=re.UNICODE)


def count_words(sentence: str) -> str:
    return len(sentence.split(" "))


notes_word_length_df = notes_text_df.applymap(remove_whitespace).applymap(count_words)
notes_word_length_df = (
    notes_word_length_df.join(notes_split_df)
    .groupby("split")
    .agg([np.median, percentile(25), percentile(75)])
).astype(int)
notes_word_length_median_df = notes_word_length_df.loc[
    :, (slice(None), "median")
].droplevel(level=1, axis="columns")
notes_word_length_percentile_25_df = notes_word_length_df.loc[
    :, (slice(None), "percentile_25")
].droplevel(level=1, axis="columns")
notes_word_length_percentile_75_df = notes_word_length_df.loc[
    :, (slice(None), "percentile_75")
].droplevel(level=1, axis="columns")

notes_word_length_table = elementwise_string_from_template(
    template="{a} ({b}, {c})",
    a=notes_word_length_median_df,
    b=notes_word_length_percentile_25_df,
    c=notes_word_length_percentile_75_df,
).T
notes_word_length_table
#%% [markdown]
# ## Combine all Tables
combined_table = pd.concat(
    [
        patient_ct_table,
        patient_ct_for_case_ct_table,
        age_stats_table,
        gender_ct_table,
        case_ct_table,
        anes_type_table,
        asa_ct_table,
        case_note_timedelta_table,
        notes_ct_table,
        notes_word_length_table,
    ]
)

tuple_for_multiindex = [
    ("Patient Count, no. (%) ", ""),
    ("Number of Surgeries per Patient, no. (%)", "1"),
    ("Number of Surgeries per Patient, no. (%)", "2"),
    ("Number of Surgeries per Patient, no. (%)", "3"),
    ("Number of Surgeries per Patient, no. (%)", "4"),
    ("Number of Surgeries per Patient, no. (%)", ">=5"),
    ("Age, mean (SD)", ""),
    ("Gender, no. (%)", "F"),
    ("Gender, no. (%)", "M"),
    ("Gender, no. (%)", "U"),
    ("Case Count, no. (%)", ""),
    ("Anesthesia Type, no. (%)", "General"),
    ("Anesthesia Type, no. (%)", "MAC"),
    ("Anesthesia Type, no. (%)", "Regional"),
    ("ASA Physical Status Classification Score, no. (%)", "I"),
    ("ASA Physical Status Classification Score, no. (%)", "II"),
    ("ASA Physical Status Classification Score, no. (%)", "III"),
    ("ASA Physical Status Classification Score, no. (%)", "IV-V"),
    ("Time Between Pre-Anesthesia Note and Surgery, median (IQR)", ""),
    ("Notes Count, no. (%)", ""),
    ("Word-Level Length, median (IQR)", "Full Note"),
    ("Word-Level Length, median (IQR)", "Procedure"),
    ("Word-Level Length, median (IQR)", "Diagnosis"),
    ("Word-Level Length, median (IQR)", "HPI"),
    ("Word-Level Length, median (IQR)", "PMSH"),
    ("Word-Level Length, median (IQR)", "ROS"),
    ("Word-Level Length, median (IQR)", "Medications"),
]
combined_table.index = pd.MultiIndex.from_tuples(tuple_for_multiindex)
combined_table
# %%
combined_table.to_csv("table1.tsv", sep="\t")
# %%
