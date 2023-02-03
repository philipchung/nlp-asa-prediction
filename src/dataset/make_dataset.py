from __future__ import annotations

import logging
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union
from urllib.parse import parse_qsl

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, DatasetInfo
from src.dataset.utils import parallel_process

log = logging.getLogger(__name__)


class MakeDataset:
    """Mix-in Class containing methods for creating Dataset for model training."""

    def __init__(self):
        MakeDataset().__init__()

    def make_hpi_asa_association(
        self, cache_path: Union[Path, str] = None
    ) -> pd.DataFrame:
        """Associate cases and notes table.  Cases with invalid ASA values and notes
        that do not have a pre-anesthesia History of Present Illness (HPI) section are excluded.

        Args:
            cache_path (Union[Path, str], optional): Path to cached result. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe that associates ProcedureID from cases table to NoteID from
            notes table.
        """
        cache_path = (
            self.interim_data_dir / "hpi-asa_association.parquet"
            if cache_path is None
            else Path(cache_path)
        )
        try:
            df = pd.read_parquet(path=cache_path)
        except:  # noqa: E722
            cases = self.cases
            notes = self.notes

            # Remove case if no ASA value
            cases = cases.loc[cases.ASA.notnull()]
            log.info("Drop Cases with no ASA Value.")
            log.info("Number of Cases: ", len(cases))

            # Only keep PreAnesthesia notes with non-empty HPI
            notes = notes.loc[notes.HPI.notnull()]
            log.info("Drop Notes with no History of Present Illness section.")
            log.info("Number of Notes: ", len(notes))

            # Associate PreAnesthesia Note to each Case
            df = self.associate_cases_and_notes(cases=cases, notes=notes)
            # Cache result
            df.to_parquet(cache_path)
        return df

    def make_pmsh_asa_association(
        self, cache_path: Union[Path, str] = None
    ) -> pd.DataFrame:
        """Associate cases and notes table.  Cases with invalid ASA values and notes
        that do not have a pre-anesthesia Past Medical/Surgical History section(s) are excluded.
        Past Medical History (PMH) and Past Surgical History (PSH) are combined into a single
        text string in this method.

        Args:
            cache_path (Union[Path, str], optional): Path to cached result. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe that associates ProcedureID from cases table to NoteID from
            notes table.
        """
        cache_path = (
            self.interim_data_dir / "pmsh-asa_association.parquet"
            if cache_path is None
            else Path(cache_path)
        )
        try:
            df = pd.read_parquet(path=cache_path)
        except:  # noqa: E722
            cases = self.cases
            notes = self.notes

            # Remove case if no ASA value
            cases = cases.loc[cases.ASA.notnull()]
            log.info("Drop Cases with no ASA Value.")
            log.info("Number of Cases: ", len(cases))

            # Only keep PreAnesthesia notes with non-empty PMH+PSH
            def _combine_pmh_psh(row: pd.Series) -> str:
                if pd.isnull(row.PMH) and pd.isnull(row.PSH):
                    return np.NaN
                else:
                    pmh = row.PMH if pd.notnull(row.PMH) else ""
                    psh = row.PSH if pd.notnull(row.PSH) else ""
                    pmsh = f"{pmh} {psh}".strip()
                    return pmsh

            notes["PMSH"] = notes.apply(_combine_pmh_psh, axis=1)
            notes = notes.loc[notes.PMSH.notnull()]
            log.info(
                "Drop Notes with no Past Medical History & Past Surgical History sections."
            )
            log.info("Number of Notes: ", len(notes))

            # Associate PreAnesthesia Note to each Case
            df = self.associate_cases_and_notes(cases=cases, notes=notes)
            # Cache result
            df.to_parquet(cache_path)
        return df

    def make_ros_asa_association(
        self, cache_path: Union[Path, str] = None
    ) -> pd.DataFrame:
        """Associate cases and notes table.  Cases with invalid ASA values and notes
        that do not have a pre-anesthesia Review of Systems (ROS) are excluded.

        Args:
            cache_path (Union[Path, str], optional): Path to cached result. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe that associates ProcedureID from cases table to NoteID from
            notes table.
        """
        cache_path = (
            self.interim_data_dir / "ros-asa_association.parquet"
            if cache_path is None
            else Path(cache_path)
        )
        try:
            df = pd.read_parquet(path=cache_path)
        except:  # noqa: E722
            cases = self.cases
            notes = self.notes

            # Remove case if no ASA value
            cases = cases.loc[cases.ASA.notnull()]
            log.info("Drop Cases with no ASA Value.")
            log.info("Number of Cases: ", len(cases))

            # Only keep PreAnesthesia notes with non-empty ROS
            notes = notes.loc[notes.ROS.notnull()]
            log.info("Drop Notes with no Review of Systems section.")
            log.info("Number of Notes: ", len(notes))

            # Associate PreAnesthesia Note to each Case
            df = self.associate_cases_and_notes(cases=cases, notes=notes)
            # Cache result
            df.to_parquet(cache_path)
        return df

    def make_meds_asa_association(
        self, cache_path: Union[Path, str] = None
    ) -> pd.DataFrame:
        """Associate cases and notes table.  Cases with invalid ASA values and notes
        that do not have a pre-anesthesia Medications section are excluded.

        Args:
            cache_path (Union[Path, str], optional): Path to cached result. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe that associates ProcedureID from cases table to NoteID from
            notes table.
        """
        cache_path = (
            self.interim_data_dir / "meds-asa_association.parquet"
            if cache_path is None
            else Path(cache_path)
        )
        try:
            df = pd.read_parquet(path=cache_path)
        except:  # noqa: E722
            cases = self.cases
            notes = self.notes

            # Remove case if no ASA value
            cases = cases.loc[cases.ASA.notnull()]
            log.info("Drop Cases with no ASA Value.")
            log.info("Number of Cases: ", len(cases))

            # Only keep PreAnesthesia notes with non-empty Medications
            notes = notes.loc[notes.Medications.notnull()]
            log.info("Drop Notes with no Medications section.")
            log.info("Number of Notes: ", len(notes))

            # Associate PreAnesthesia Note to each Case
            df = self.associate_cases_and_notes(cases=cases, notes=notes)
            # Cache result
            df.to_parquet(cache_path)
        return df

    def make_hpi_pmsh_ros_meds_asa_association(
        self, cache_path: Union[Path, str] = None
    ) -> pd.DataFrame:
        """Associate cases and notes table.  Cases with invalid ASA values and notes
        that do not have a pre-anesthesia sections are excluded.

        Args:
            cache_path (Union[Path, str], optional): Path to cached result. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe that associates ProcedureID from cases table to NoteID from
            notes table.
        """
        cache_path = (
            self.interim_data_dir / "hpi-pmsh-ros-meds-asa_association.parquet"
            if cache_path is None
            else Path(cache_path)
        )
        try:
            df = pd.read_parquet(path=cache_path)
        except:  # noqa: E722
            cases = self.cases
            notes = self.notes

            # Remove case if no value for ASA, ProcedureDescription, DiagnosisDescription
            cases = cases.loc[
                cases.ASA.notnull()
                & cases.ProcedureDescription.notnull()
                & cases.DiagnosisDescription.notnull()
            ]
            log.info(
                "Drop Cases with missing: ASA, ProcedureDescription, DiagnosisDescription."
            )
            log.info("Number of Cases: ", len(cases))

            # Only keep PreAnesthesia notes with non-empty NoteText, HPI, PMSH, ROS, Medications
            notes = notes.loc[
                notes.NoteText.notnull()
                & notes.HPI.notnull()
                & notes.PMSH.notnull()
                & notes.ROS.notnull()
                & notes.Medications.notnull()
            ]
            log.info("Drop Notes with missing: NoteText, HPI, PMSH, ROS, Meds.")
            log.info("Number of Notes: ", len(notes))

            # Associate PreAnesthesia Note to each Case
            df = self.associate_cases_and_notes(cases=cases, notes=notes)
            # Cache result
            df.to_parquet(cache_path)
        return df

    def make_dataframe(self, case_note_association: pd.DataFrame) -> pd.DataFrame:
        """Given `case_note_association` which links each case ProcedureID to
        a NoteID, join both cases and notes table to expand this association to
        a full dataframe with columns from both tables.

        Note: the case_note_association is computationally expensive to compute and the
        result is cached for future use.

        Args:
            case_note_association (pd.DataFrame): A dataframe of shape (n,1) which
                associates a ProcedureID (row index) to a NoteID (column value).

        Returns:
            pd.DataFrame: Dataframe with each row corresponding to a unique ProcedureID
                and columns from both cases and notes table where data from notes table
                are from the associated row that corresponds to NoteID.
        """
        df = case_note_association.join(
            self.cases.drop(columns="EncounterID"), how="left"
        )
        df = (
            pd.merge(
                df,
                self.notes.drop(columns=["PersonID", "EncounterID"]),
                left_on="PreAnesthesiaNoteID",
                right_index=True,
                how="left",
            )
            .reset_index()
            .rename_axis(index="index")
        )
        return df

    def format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        "Selects a subset of columns."
        # Reorder Columns based on Categorization of Variables as Input vs. Output Features
        info_columns = ["PersonID", "ProcedureID", "PreAnesthesiaNoteID"]
        input_columns = [
            "Age",
            "Gender",
            "AnesType",
            "PatientClass",
            "ProcedureDescription",
            "ProcedureType",
            "DiagnosisDescription",
            "VisitType",
            "Procedure",
            "Diagnosis",
            "RCRI",
            "TotalRCRI",
            "HPI",
            "ROS",
            "PMH",
            "PSH",
            "PMSH",
            "Medications",
            "NoteText",
        ]
        output_columns = [
            "ASA",
        ]
        all_columns = info_columns + input_columns + output_columns
        df = df.loc[:, all_columns]
        return df

    def make_train_validation_test_datasets(
        self,
        df: pd.DataFrame = None,
        seed: int = 1,
        split: tuple[float, float, float] = (0.7, 0.1, 0.2),
        license_path: Union[str, Path] = None,
    ) -> DatasetDict:
        """
        Create dataset with train, validation, test splits.
        If patient has multiple cases, all of those patient's cases
        will reside within a single split to avoid data leakage.
        """
        if sum(split) != 1.0:
            raise ValueError(
                "Arg `split` must be tuple with 3 elements that sum to 1.0."
            )
        # Get all unique PersonID
        person_ids = df.PersonID.unique()
        # Shuffle
        rng = np.random.default_rng(seed=seed)
        rng.shuffle(person_ids)
        # 70-10-20 Train/Validation/Test split based on unique PersonID
        train_split_index = int(len(person_ids) * split[0])
        val_split_index = int(len(person_ids) * (split[0] + split[1]))
        train_person_ids, validation_person_ids, test_person_ids = np.split(
            person_ids, [train_split_index, val_split_index]
        )

        train_df = df.loc[df.PersonID.isin(train_person_ids)]
        validation_df = df.loc[df.PersonID.isin(validation_person_ids)]
        test_df = df.loc[df.PersonID.isin(test_person_ids)]

        # Dataset Info
        _CITATION = """\
        @misc{philipchung:{self.dataset_name},
            title = {self.dataset_name},
            authors = {Philip Chung},
            year = {2022}
        }
        """

        _DESCRIPTION = """\
        PreSurgeryNLP Dataset contains perioperative and surgical info \
        abstracted from medical record from Jan 2012 - March 2021.  \
        It contains clinical texts of notes for patients prior to surgery, \
        descriptions and information about the surgery, \
        and ASA Physical Status Classification assigned on day of surgery.
        """

        _HOMEPAGE = "https://github.com/philipchung/nlp-asa-prediction/"

        license_path = (
            Path(self.project_dir / "LICENSE") if license_path is None else license_path
        )
        with open(file=license_path) as license_file:
            _LICENSE = license_file.read()

        info = DatasetInfo(
            description=_DESCRIPTION,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

        # Make datasets
        train_dataset = Dataset.from_pandas(df=train_df, info=info).shuffle(seed=seed)
        validation_dataset = Dataset.from_pandas(df=validation_df, info=info).shuffle(
            seed=seed
        )
        test_dataset = Dataset.from_pandas(df=test_df, info=info).shuffle(seed=seed)

        dataset_dict = DatasetDict(
            {
                "train": train_dataset,
                "validation": validation_dataset,
                "test": test_dataset,
            }
        )
        return dataset_dict

    def associate_cases_and_notes(
        self,
        cases: pd.DataFrame = None,
        notes: pd.DataFrame = None,
        grace_period_to_file_note: int = 6,
        max_days_between_note_and_surgery: int = 90,
    ) -> pd.DataFrame:
        """
        Joins Cases and Notes table.

        Args:
            cases (pd.DataFrame): dataframe of cases
            notes (pd.DataFrame): dataframe of notes
            grace_period_to_file_note: hours after surgery that pre-anesthesia note must be filed by
            max_days_between_note_and_surgery: max days before surgery that pre-anesthesia note
                must be filed by

        Returns:
            dataframe with index `ProcedureID` from cases dataframe and single column
            of `PreAnesthesiaNoteID` from notes dataframe.  This defines which
            preanesthesia note is associated with a surgery case.
        """
        cases = self.cases if cases is None else cases
        notes = self.notes if notes is None else notes
        # Create a subset of Notes & Cases dataframe with only TimeStamp info
        # to minimize the amount of data that must be pickled and sent via parallel_process.
        # Convert cases to a dict to iterate over it faster.
        notes_subset = notes.loc[:, ["PersonID", "LastUpdateDateTime"]].reset_index()
        cases_subset = cases.loc[:, ["PersonID", "AnesStart", "AnesEnd"]].reset_index()
        cases_dict = cases_subset.to_dict("records")
        # Sometimes these notes are filed after surgery end,
        # so we put cut-off at 6 hrs after surgery (unlikely to have repeat surgery by then).
        # We also want relatively recent preanesthesia notes and require it to be
        # filed within the last 90 days.
        associate_note_and_case = partial(
            associate_case_to_preanesthesia_note,
            notes=notes_subset,
            grace_period_to_file_note=grace_period_to_file_note,
            max_days_between_note_and_surgery=max_days_between_note_and_surgery,
        )
        # This step can take a long time
        case_note_assoc = parallel_process(
            iterable=cases_dict,
            function=associate_note_and_case,
            n_jobs=cpu_count(),
            desc="Associate Cases and Notes",
        )
        case_note_df = pd.DataFrame(case_note_assoc).set_index("ProcedureID")
        # Remove cases where there is no HPI we can associate with it
        mask = case_note_df.PreAnesthesiaNoteID.apply(
            lambda x: True if x is not None else False
        )
        case_note_df = case_note_df.loc[mask]
        print(
            "For each case, associate pre-anesthesia note.  "
            "Drop cases with no associable pre-anesthesia note."
        )
        print("Number of Cases: ", len(case_note_df))
        return case_note_df


def associate_case_to_preanesthesia_note(
    case_dict: dict,
    notes: pd.DataFrame,
    grace_period_to_file_note: int,
    max_days_between_note_and_surgery: int,
) -> dict:
    """
    Given case_dict (row of cases table), associate preanesthesia NoteID.

    Args:
        case_dict: cases dataframe converted to dictionary records.  We iterate over these records.
        notes: notes dataframe
        grace_period_to_file_note: hours after surgery that pre-anesthesia note must be filed by
        max_days_between_note_and_surgery: max days before surgery that pre-anesthesia note must be filed by

    Returns:
        Dict of case ProcedureID and associated PreAnesthesiaNoteID
    """
    procedure_id = case_dict["ProcedureID"]
    person_id = case_dict["PersonID"]
    surgery_start = case_dict["AnesStart"]
    surgery_end = case_dict["AnesEnd"]
    # Get preanesthesia notes for patient
    n = notes.loc[notes.PersonID == person_id]
    if n.empty:
        return {"ProcedureID": procedure_id, "PreAnesthesiaNoteID": None}
    else:
        # Get preanesthesia notes written prior to surgery.
        preanes_note = n.loc[
            (
                n.LastUpdateDateTime
                < (surgery_end + pd.Timedelta(hours=grace_period_to_file_note))
            )
            & (
                (surgery_start - n.LastUpdateDateTime)
                <= pd.Timedelta(days=max_days_between_note_and_surgery)
            )
        ]
        if preanes_note.empty:
            return {"ProcedureID": procedure_id, "PreAnesthesiaNoteID": None}
        else:
            # Associate most recent preanesthesia note
            preanes_note = preanes_note.nlargest(n=1, columns="LastUpdateDateTime")
            preanes_note_id = preanes_note.NoteID.iloc[0]
            return {
                "ProcedureID": procedure_id,
                "PreAnesthesiaNoteID": preanes_note_id,
            }
