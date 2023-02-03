import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union
from .utils import find_pattern
from .preanesthesia_note import (
    segment_preanesthesia_notes_to_sections,
    extract_variables_from_segmented_notes,
)


class Notes:
    """Mix-in Class containing methods for processing Notes data."""

    def __init__(
        self,
        notes: pd.DataFrame = None,
    ):
        self.notes = notes

    def clean_notes(self, notes: pd.DataFrame = None) -> pd.DataFrame:
        "Convert timestamp to python datetime."
        notes = self.notes if notes is None else notes

        notes.LastUpdateDateTime = notes.LastUpdateDateTime.apply(
            lambda x: x.to_pydatetime()
        )
        return notes

    def is_preanesthesia(
        self, notes: pd.DataFrame = None, cache_path: Union[Path, str] = None
    ) -> pd.DataFrame:
        "Find valid preanesthesia notes, join annotation to notes dataframe, cache result."
        notes = self.notes if notes is None else notes

        try:
            is_preanesthesia = pd.read_parquet(path=cache_path)
        except:  # noqa: E722
            is_preanesthesia = self._is_preanesthesia(notes)
            Path(cache_path.parent).mkdir(parents=True, exist_ok=True)
            is_preanesthesia.to_frame().to_parquet(path=cache_path)

        notes = notes.join(is_preanesthesia)
        return notes

    def _is_preanesthesia(self, notes: pd.DataFrame) -> pd.DataFrame:
        """Filter valid pre-anesthesia notes by NoteText contents.
        Many notes are mis-categorized, so we cannot rely on metadata to identify a PreAnesthesia Note.
        Instead, we use regex to match string text in notes based on phenotype.
        Args:
            notes: notes dataframe
        Returns:
            Filtered notes dataframe with only pre-anesthesia notes based on phenotype.
        """

        def _preanesthesia_pattern(text: str) -> bool:
            "Determines based on text features (PowerNote template) whether note is a PreAnesthesia Note."
            match = find_pattern(
                text, pattern=r" \n \n \nVISIT INFORMATION: \nOR Date: "
            )
            return True if match else False

        tqdm.pandas(desc="Finding PreAnesthesia Notes")
        return notes.NoteText.progress_apply(_preanesthesia_pattern).rename(
            "IsPreAnesthesia"
        )

    def segment_preanesthesia_notes(
        self, notes: pd.DataFrame = None, cache_path: Union[Path, str] = None
    ) -> pd.DataFrame:
        notes = self.notes if notes is None else notes

        try:
            extracted = pd.read_parquet(path=cache_path)
        except:  # noqa: E722
            extracted = self._segment_preanesthesia_notes(notes)

            # Serialize Enum Types into String for Arrow Compatibility
            tqdm.pandas(desc="VisitType enum -> string", dynamic_ncols=True)
            extracted.VisitType = extracted.VisitType.progress_apply(
                lambda x: x.serialize() if pd.notnull(x) else None
            )
            tqdm.pandas(desc="RCRI enum -> string", dynamic_ncols=True)
            extracted.RCRI = extracted.RCRI.progress_apply(
                lambda x: x.serialize() if pd.notnull(x) else None
            )

            Path(cache_path.parent).mkdir(parents=True, exist_ok=True)
            extracted.to_parquet(path=cache_path)

        notes = notes.join(extracted)
        return notes

    def _segment_preanesthesia_notes(self, notes: pd.DataFrame) -> pd.DataFrame:
        """Extracts fields out of free-text using regular expression pattern matching."""
        # Get only Pre-Anesthesia Notes
        notes = notes.loc[notes.IsPreAnesthesia]

        segmented_notes = segment_preanesthesia_notes_to_sections(notes.NoteText)
        extracted = extract_variables_from_segmented_notes(segmented_notes)
        return extracted

    def make_pmsh(self, notes: pd.DataFrame) -> pd.DataFrame:
        """Combines PMH and PSH into single column PMSH."""
        notes = self.notes if notes is None else notes

        # Combine PMH and PSH from segmented notes to make PMSH column
        def _combine_pmh_psh(row: pd.Series) -> str:
            if pd.isnull(row.PMH) and pd.isnull(row.PSH):
                return np.NaN
            else:
                pmh = row.PMH if pd.notnull(row.PMH) else ""
                psh = row.PSH if pd.notnull(row.PSH) else ""
                pmsh = f"{pmh} {psh}".strip()
                return pmsh

        notes["PMSH"] = notes.apply(_combine_pmh_psh, axis=1)
        return notes
