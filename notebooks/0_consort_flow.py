#%% [markdown]
# ## Dataset Numbers for Consort Flow Diagram
#%%
from src.modeling.datamodule import DataModule
from src.dataset.data import Data
from pathlib import Path
import pandas as pd
import logging

project_root_dir = Path(__file__).parent.parent

logger = logging.getLogger("dataset_logger")
logger.setLevel(logging.INFO)

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
validation_df = datamodule.val_dataframe()
test_df = datamodule.test_dataframe()
print(f"Final Train Shape: {train_df.shape}")
print(f"Final Validation Shape: {validation_df.shape}")
print(f"Final Test Shape: {test_df.shape}")
# Table of Associated Surgery Cases & Notes
all_df = pd.concat([train_df, validation_df, test_df], axis=0)
print(f"Final Dataframe (Train+Validation+Test) Shape: {all_df.shape}")

#%%
# Load Raw Data
data = Data(
    project_dir=project_root_dir,
    datastore_blob_path=datamodule.data_blob_prefix,
    dataset_name=datamodule.dataset_name,
    seed=datamodule.seed,
)
cases = data.get_cases()
notes = data.get_notes()
print(f"Raw Cases Shape: {cases.shape}")
print(f"Raw Notes Shape: {notes.shape}")

#%%
# Raw & Intermediate Note Data
raw_cases = pd.read_parquet(data.raw_data_dir / "cases.parquet")
filtered_cases = pd.read_parquet(data.interim_data_dir / "filtered_cases.parquet")
merge_overlap_cases = pd.read_parquet(data.interim_data_dir / "merged_cases.parquet")
raw_cases.shape, filtered_cases.shape, merge_overlap_cases.shape
#%%
# Raw & Intermediate Note Data
raw_notes = pd.read_parquet(data.raw_data_dir / "notes.parquet")
is_preanesthesa_notes = pd.read_parquet(
    data.interim_data_dir / "is_preanesthesia.parquet"
)
extracted_preanesthesia_notes = pd.read_parquet(
    data.interim_data_dir / "preanesthesia_extracted.parquet"
)
raw_notes.shape, is_preanesthesa_notes.shape, extracted_preanesthesia_notes.shape
# %%
# Remove case if no value for ASA, ProcedureDescription, DiagnosisDescription
preassociation_cases = cases.loc[
    cases.ASA.notnull()
    & cases.ProcedureDescription.notnull()
    & cases.DiagnosisDescription.notnull()
]
print("Transformation of Raw Cases -> Filtered Cases")
print("Drop Cases with missing: ASA, ProcedureDescription, DiagnosisDescription.")
print("Number of Cases: ", len(preassociation_cases))

# Only keep PreAnesthesia notes with non-empty NoteText, HPI, PMSH, ROS, Medications
preassociation_notes = notes.loc[
    notes.NoteText.notnull()
    & notes.HPI.notnull()
    & notes.PMSH.notnull()
    & notes.ROS.notnull()
    & notes.Medications.notnull()
]
print("Transformation of Raw Notes -> Filtered Notes")
print("Drop Notes with missing: NoteText, HPI, PMSH, ROS, Meds.")
print("Number of Notes: ", len(preassociation_notes))

#%%


#%%
# Raw Cases
# -> Filter cases -> Filtered Cases
# -> Merge Overlap Cases -> Merged Cases
# -> Drop Case without ASA, ProcedureDescription, DiagnosisDescription -> Pre-Association Cases
print(
    f"""Raw Cases: {cases.shape} 
    -> Filtered Cases: {filtered_cases.shape} 
    -> Merge Overlap Cases: {merge_overlap_cases.shape} 
    -> Pre-Association Cases: {preassociation_cases.shape}"""
)

# Raw Notes
# -> Determine PreAnesthesia -> Preanesthesia Notes
# -> Drop Blank Notes, Drop Notes with Missing HPI, PMSH, ROS, Medications -> Preassociation Notes
print(
    f"""Raw Notes: {notes.shape} 
    -> PreAnesthesia Notes: {extracted_preanesthesia_notes.shape} 
    -> PreAssociation Notes: {preassociation_notes.shape}"""
)
#%%
# For each patient, Filtered Cases & Filter Notes are associated based on 2 criteria:
#   - Criteria 1: Each case is associated with most recent note written prior to anesthesia start based on note file time
#       - Allow grace period for note to be filed upto 6 hours after anesthesia end
#       (to account for cases where note is not written in advance or modified and filed at case end)
#       (We are extremely unlikely to have repeat surgery within 6 hours time)
#   - Criteria 2: Only notes written within last 90 days of surgery are considered.  Older notes may not
#       reflect the patient's current medical situation.
# Result is a table of cases & notes
print(f"After Case-Note Association: {all_df.shape}")
# Train, Validation, Test Splits are generated with all of a patient's surgeries randomized into a single split to avoid data leakage
print(f"Final Train Shape: {train_df.shape}")
print(f"Final Validation Shape: {validation_df.shape}")
print(f"Final Test Shape: {test_df.shape}")
# %%
