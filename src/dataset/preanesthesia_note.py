from __future__ import annotations

import re
from collections import namedtuple
from functools import partial
from typing import Union

import pandas as pd

from src.dataset.enums import RCRI, VisitType
from src.dataset.utils import extract_field
from src.dataset.utils import parallel_process

Segment = namedtuple("Segment", ["name", "start", "end"])


def note_segmenter(text: str, sections: list = None) -> list[Segment]:
    """Segment the sections of a PreAnesthesia PowerNote.
    Generates segments that identify spans in text, splitting up by `sections`
    and also makes a distinction between section header & body.  Resulting spans
    can be used to index into original text to retrieve the text segment.
    Args:
        text: note text that will be segmented
        sections: section headers used to segment text (case insensitive)
    Returns:
        Dictionary of {sections: re.match obj corresponding to section}"""
    if not sections:
        sections = [
            "VISIT INFORMATION",
            "HISTORY OF PRESENT ILLNESS",
            "HISTORY SOURCE",
            "PAST MEDICAL HISTORY",
            "PAST SURGICAL HISTORY",
            "FAMILY ANESTHESIA HISTORY",
            "SOCIAL BEHAVIORS",
            "MEDICATIONS",
            "ALLERGIES",
            "REVIEW OF SYSTEMS",
            "ADVANCE DIRECTIVES",
            "PHYSICAL EXAMINATION",
            "RESULTS REVIEW",
            "PROCEDURES/DIAGNOSTIC STUDIES",
            "SUMMARY/RECOMMENDATIONS",
            "PATIENT EDUCATION",
        ]
    # Find all section headers
    sect_pattern = [f"({sect}: \n)" for sect in sections]
    any_section_pattern = "|".join(sect_pattern)
    headers_match = re.finditer(any_section_pattern, text)
    headers_match = list(headers_match) if headers_match else []
    # note: headers are matched in-order of appearance in text

    # Define Segments using Spans Indexing into original text
    pre_seg = Segment("pre", 0, headers_match[0].start())
    segments = [pre_seg]

    for i, match in enumerate(headers_match):
        # Header Span
        header_start, header_end = match.start(), match.end()
        header_name = match.group().strip().strip(":")
        # Index to where body stops
        if match != headers_match[-1]:
            next_header_start = headers_match[i + 1].start()
        else:
            next_header_start = len(text)
        # Header Segment
        header_seg = Segment(header_name, header_start, header_end)
        segments.append(header_seg)
        # Body Text Segment (between 2 headers)
        body_seg = Segment(header_name + "_body", header_end, next_header_start)
        segments.append(body_seg)

    # If first header segment starts at 0, then drop "pre" segment
    while segments[1].start == 0:
        segments = segments[1:]
    return segments


def _spans_to_text(text: str, spans: list) -> dict:
    "Uses spans as start/stop indicies to extract text segments."
    output = {}
    for segment in spans:
        if segment:
            output[segment.name] = text[segment.start : segment.end]
        else:
            output[segment.name] = None
    return output


def segment_preanesthesia_notes_to_sections(notes_series: pd.Series) -> pd.DataFrame:
    """Detects sections within notes based on `note_segmenter()` function,
    and returns text for each segment.
    Args:
        notes_series: pandas series of note text.  Notes must follow template
            of `Anesthesia PreOperative Assessment`
    Returns:
        Dataframe where each note is segmented and resultant text segments are
        under column labeled by section name. `note_segmenter()` will break up
        section header and section body text into separate columns.  Section
        header column name is section name as defined in `note_segmenter()`.
        Section body column name is header name with "_body" appended to it.
        (e.g. "VISIT INFORMATION" is column name for header text,
        "VISIT INFORMATION_body" is column name for body text)
    """

    # Identify note segments as numerical spans
    segment_spans = parallel_process(
        iterable=notes_series, function=note_segmenter, desc="Segmenting Notes"
    )

    # Extract actual text segments
    notes_sections = parallel_process(
        iterable=zip(notes_series, segment_spans),
        function=_spans_to_text,
        use_args=True,
        desc="Extract Text Segments",
    )
    notes_sections = pd.DataFrame(notes_sections, index=notes_series.index)
    return notes_sections


#%% [markdown]
### Functions for Extracting Variables from Segmented Notes
#%%
def extract_visit_info(text: str) -> dict[str, Union[VisitType, str]]:
    """Extracts variables from VISIT INFORMATION section of preanesthesia note.
    Args:
        text: string of VISIT INFORMATION body section of note
    Returns:
        Dict with keys "VisitType", "Procedure", "Diagnosis", and values
        of the corresponding extracted variables.  If empty string passed
        as `text` argument, then values will be `NaN`.
    """
    if not bool(text):
        return {"VisitType": VisitType(0), "Procedure": None, "Diagnosis": None}
    else:
        extract_visit_type = partial(extract_field, pre="\nType of Visit: ", post="\n")
        extract_procedure = partial(
            extract_field, pre="\nProposed Procedure: ", post="\n"
        )
        extract_diagnosis = partial(extract_field, pre="\nDiagnosis: ", post="\n")

        # Extract visit type as Enum.  If multiple visit type, reduces down to one.
        visit_type = extract_visit_type(text)
        visit_type = (
            VisitType.from_string(visit_type[0].text).reduce()
            if visit_type
            else VisitType(0)
        )

        procedure = extract_procedure(text)
        procedure = procedure[0].text if procedure else None

        diagnosis = extract_diagnosis(text)
        diagnosis = diagnosis[0].text if diagnosis else None
        return {"VisitType": visit_type, "Procedure": procedure, "Diagnosis": diagnosis}


def extract_review_of_systems(text: str) -> dict[str, Union[RCRI, int]]:
    """Extracts variables from REVIEW OF SYSTEMS section of preanesthesia note.
    Args:
        text: string of REVIEW OF SYSTEMS body section of note
    Returns:
        Dict with keys "RCRI", "TotalRCRI", "ROS", and values
        of the corresponding extracted variables.  If empty string passed
        as `text` argument, then values will be `NaN`.
    """
    if not bool(text):
        return {"RCRI": RCRI(0), "TotalRCRI": 0, "ROS": None}
    else:
        extract_rcri = partial(
            extract_field, pre="\nRevised Cardiac Risk Index: ", post="\n"
        )
        # Extract RCRI as Enum.
        rcri = extract_rcri(text)
        rcri = RCRI.from_string(rcri[0].text) if rcri else RCRI(0)
        total_rcri = rcri.count_flags()
        return {"RCRI": rcri, "TotalRCRI": total_rcri, "ROS": text}


def extract_history_of_present_illness(text: str) -> dict[str, str]:
    """Extracts variables from HISTORY OF PRESENT ILLNESS section of preanesthesia note.
    Args:
        text: string of HISTORY OF PRESENT ILLNESS body section of note
    Returns:
        Dict with key "HPI", and value of extracted HPI text.  If empty string passed
        as `text` argument, then values will be `NaN`.
    """
    return {"HPI": text if bool(text) else None}


def extract_past_medical_history(text: str) -> dict[str, str]:
    """Extracts variables from PAST MEDICAL HISTORY section of preanesthesia note.
    Args:
        text: string of PAST MEDICAL HISTORY body section of note
    Returns:
        Dict with key "PMH", and value of extracted PMH text.  If empty string passed
        as `text` argument, then values will be `NaN`.
    """
    return {"PMH": text if bool(text) else None}


def extract_past_surgical_history(text: str) -> dict[str, str]:
    """Extracts variables from PAST SURGICAL HISTORY section of preanesthesia note.
    Args:
        text: string of PAST SURGICAL HISTORY body section of note
    Returns:
        Dict with key "PSH", and value of extracted PSH text.  If empty string passed
        as `text` argument, then values will be `NaN`.
    """
    return {"PSH": text if bool(text) else None}


def extract_medications(text: str) -> dict[str, str]:
    """Extracts variables from MEDICATIONS section of preanesthesia note.
    Args:
        text: string of MEDICATIONS body section of note
    Returns:
        Dict with key "Medications", and value of extracted Medications text.  If empty string passed
        as `text` argument, then values will be `NaN`.
    """
    return {"Medications": text if bool(text) else None}


def extract_variables_from_segmented_notes(
    segmented_notes: pd.DataFrame,
) -> pd.DataFrame:
    """Extracts variables of interest from preanesthesia note.
    Args:
        segmented_notes: pandas Dataframe of preanesthesia note with
            each section and section header split out in its own column.
            This is achieved by calling the function
            segment_preanesthesia_notes_by_section() on a Series of individual notes.
    Returns:
        Pandas dataframe with extracted variables for each note.
        Columns include "VisitType", "Procedure", "Diagnosis", "RCRI",
        "TotalRCRI", "HPI", "PMH", "PSH", "Medications", "ROS".
        Index for resultant Pandas dataframe will be the same as the input `notes` dataframe.
    """
    visit_info, ros, hpi, pmh, psh, medications = [None] * 6

    if "VISIT INFORMATION_body" in segmented_notes.columns:
        visit_info = parallel_process(
            iterable=segmented_notes.loc[:, "VISIT INFORMATION_body"],
            function=extract_visit_info,
            desc="Extract Visit Info",
        )
        visit_info = pd.DataFrame(visit_info)

    if "REVIEW OF SYSTEMS_body" in segmented_notes.columns:
        ros = parallel_process(
            iterable=segmented_notes.loc[:, "REVIEW OF SYSTEMS_body"],
            function=extract_review_of_systems,
            desc="Extract Review of Systems",
        )
        ros = pd.DataFrame(ros)

    if "HISTORY OF PRESENT ILLNESS_body" in segmented_notes.columns:
        hpi = parallel_process(
            iterable=segmented_notes.loc[:, "HISTORY OF PRESENT ILLNESS_body"],
            function=extract_history_of_present_illness,
            desc="Extract History of Present Illness",
        )
        hpi = pd.DataFrame(hpi)

    if "PAST MEDICAL HISTORY_body" in segmented_notes.columns:
        pmh = parallel_process(
            iterable=segmented_notes.loc[:, "PAST MEDICAL HISTORY_body"],
            function=extract_past_medical_history,
            desc="Extract Past Medical History",
        )
        pmh = pd.DataFrame(pmh)

    if "PAST SURGICAL HISTORY_body" in segmented_notes.columns:
        psh = parallel_process(
            iterable=segmented_notes.loc[:, "PAST SURGICAL HISTORY_body"],
            function=extract_past_surgical_history,
            desc="Extract Past Surgical History",
        )
        psh = pd.DataFrame(psh)

    if "MEDICATIONS_body" in segmented_notes.columns:
        medications = parallel_process(
            iterable=segmented_notes.loc[:, "MEDICATIONS_body"],
            function=extract_medications,
            desc="Extract Medications",
        )
        medications = pd.DataFrame(medications)

    extracted_df = [visit_info, ros, hpi, pmh, psh, medications]
    extracted_df = pd.concat(extracted_df, axis=1).set_index(segmented_notes.index)
    return extracted_df
