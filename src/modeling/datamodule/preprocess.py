from __future__ import annotations

import logging
import os
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union

import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .dataset_utils import map_dataset_dict_with_fingerprint, select_dataset_columns

logger = logging.getLogger("dataset_logger")


# Note: datasets has built-in caching mechanism by computing fingerprint hashes.
# This only works if transforms are pickleable.  Transforms are intentionally kept
# as top-level functions outside of classes in order to help ensure they are pickleable.


def preprocess_dataset(
    dataset_path: Union[str, Path] = None,
    task_name: str = None,
    tokenizer: Union[AutoTokenizer, PreTrainedTokenizerFast] = None,
    seq_max_length: int = 512,
    asa_label2id: dict = None,
    parallel: bool = False,
) -> Union[Dataset, DatasetDict]:
    """
    Load and run dataset transformation pipeline.

    Args:
        dataset_path (str, Path): path to .arrow dataset or dataset_dict
        task_name (str): target modeling task.  Valid values include
            {"procedure-asa", "diagnosis-asa", "hpi-asa", "ros-asa", "pmsh-asa",
            "meds-asa", "note-asa", "note512-asa"}
        tokenizer (huggingface tokenizer): used to tokenize input text
        seq_max_length (int): max length of input text
        asa_label2id (dict): mapping of output label to numerical id

    Returns:
        Processed Dataset or DatasetDict.
    """
    if task_name is None:
        raise ValueError("Must specify target modeling task for dataset.")
    ds = datasets.load_from_disk(Path(dataset_path).as_posix())
    ds = transform_dataset(
        ds=ds,
        task=task_name,
        tokenizer=tokenizer,
        asa_label2id=asa_label2id,
        seq_max_length=seq_max_length,
        static_fingerprints=False,
        num_proc=cpu_count() if parallel else 1,
    )
    return ds


def select_dataset_subset_for_task(
    ds: Union[Dataset, DatasetDict], task: str
) -> Union[Dataset, DatasetDict]:
    "Select subset of dataset columns for specific prediction task."
    index_cols = [
        "index",
        "PersonID",
        "ProcedureID",
        "PreAnesthesiaNoteID",
    ]
    cols_for_task = {
        "procedure-asa": index_cols + ["ProcedureDescription", "ASA"],
        "diagnosis-asa": index_cols + ["DiagnosisDescription", "ASA"],
        "hpi-asa": index_cols + ["HPI", "ASA"],
        "ros-asa": index_cols + ["ROS", "ASA"],
        "pmsh-asa": index_cols + ["PMSH", "ASA"],
        "meds-asa": index_cols + ["Medications", "ASA"],
        "note-asa": index_cols + ["NoteText", "ASA"],
        "note512-asa": index_cols + ["NoteText", "ASA"],
    }
    output_ds = select_dataset_columns(dataset=ds, columns=cols_for_task[task])
    return output_ds


def prepare_tokenizer(
    tokenizer: Union[AutoTokenizer, PreTrainedTokenizerFast] = None,
    seq_max_length: int = 512,
) -> Union[AutoTokenizer, PreTrainedTokenizerFast]:
    "Invoke tokenizer truncation mode, otherwise will not cache results correctly."
    # Disable Tokenizers Parallelism which clashes with Dataloader Worker Parallelism
    # https://stackoverflow.com/a/72926996
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Call Tokenizer on Dummy Example to fix Tokenizer Caching Issue
    # https://github.com/huggingface/datasets/issues/3638
    tokenizer(
        "dummy text used to fix tokenizer caching issue",
        truncation=True,
        padding="max_length",
        max_length=seq_max_length,
    )
    return tokenizer


def transform_dataset_inputs(
    ds: Union[Dataset, DatasetDict],
    input_feature_name: str = None,
    tokenizer: Union[AutoTokenizer, PreTrainedTokenizerFast] = None,
    seq_max_length: int = 512,
    static_fingerprints: bool = False,
    num_proc: int = 4,
) -> Union[Dataset, DatasetDict]:
    "Select input variables, apply data transformation e.g. tokenization."

    def select_input_text(
        batch: dict[str, list], input_feature_name: str
    ) -> dict[str, list]:
        """Select input text feature.
        All further input processing is performed on `input_text` column.
        """
        batch["input_text"] = [x for x in batch[input_feature_name]]
        return batch

    fn = partial(select_input_text, input_feature_name=input_feature_name)

    name = select_input_text.__name__
    fingerprint = f"{input_feature_name}-{name}"
    ds = map_dataset_dict_with_fingerprint(
        dataset_dict=ds,
        fingerprint_base=fingerprint if static_fingerprints else None,
        function=fn,
        desc=name,
        batched=True,
        num_proc=num_proc,
    )

    def clean_whitespace(batch: dict[str, list]) -> dict[str, list]:
        "Remove newline tokens & duplicate whitespace."
        input_text = []
        for text in batch["input_text"]:
            text = text.replace("\n", " ")  # Remove newline tokens
            text = " ".join(text.split())  # Remove duplicate & trailing whitespace
            input_text += [text]
        batch["input_text"] = input_text
        return batch

    name = clean_whitespace.__name__
    fingerprint = f"{input_feature_name}-{name}"
    ds = map_dataset_dict_with_fingerprint(
        dataset_dict=ds,
        fingerprint_base=fingerprint if static_fingerprints else None,
        function=clean_whitespace,
        desc=name,
        batched=True,
        num_proc=num_proc,
    )

    # Tokenize input text
    # Note: when calling datasets.map() on this function, set num_proc=1
    # to avoid tokenizer parallelism deadlock
    tokenize_fn = lambda batch: tokenizer(
        batch["input_text"],
        truncation=True,
        padding="max_length",
        max_length=seq_max_length,
    )
    name = "tokenize_text"
    fingerprint = f"{input_feature_name}-{name}"
    ds = map_dataset_dict_with_fingerprint(
        dataset_dict=ds,
        fingerprint_base=fingerprint if static_fingerprints else None,
        function=tokenize_fn,
        desc=name,
        batched=True,
        num_proc=1,
    )
    return ds


def make_truncated_note_text(
    ds: Union[Dataset, DatasetDict],
    input_feature_name: str = None,
    tokenizer: Union[AutoTokenizer, PreTrainedTokenizerFast] = None,
    seq_max_length: int = 512,
    static_fingerprints: bool = False,
    num_proc: int = 4,
) -> Union[Dataset, DatasetDict]:
    "Make New Feature Column `TruncatedNoteText`."

    def decode_input_ids_to_truncated_note_feature(
        batch: dict[str, list]
    ) -> dict[str, list]:
        "Decode input ids to text and set as new feature column `TruncatedNoteText`."
        decoded_text = tokenizer.batch_decode(
            batch["input_ids"],
            truncation=True,
            padding="max_length",
            max_length=seq_max_length,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        batch["TruncatedNoteText"] = decoded_text
        batch["input_text"] = decoded_text
        return batch

    name = "token_decode"
    fingerprint = f"{input_feature_name}-{name}"
    ds = map_dataset_dict_with_fingerprint(
        dataset_dict=ds,
        fingerprint_base=fingerprint if static_fingerprints else None,
        function=decode_input_ids_to_truncated_note_feature,
        desc=name,
        batched=True,
        num_proc=1,
    )
    return ds


def transform_dataset_outputs(
    ds: Union[Dataset, DatasetDict],
    asa_label2id: dict = None,
    static_fingerprints: bool = False,
    num_proc: int = 4,
) -> Union[Dataset, DatasetDict]:
    "Apply data transformation to ASA Physical Status Classification output variable."

    logger.info(
        f"""Before transform outputs.
        Shape: {ds.shape} 
        Num Samples: {sum(ds.num_rows.values())}"""
    )

    def split_asa_feature(batch: dict[str, list]):
        "Split out emergency status."
        asa, emergency = [], []
        for x in batch["ASA"]:
            x = x.upper()
            # Split out Emergent "E" designation out of ASA status
            if x.endswith("E"):
                emergency_status = True
                x = x.rstrip("E")
            else:
                emergency_status = False
            asa += [x]
            emergency += [emergency_status]
        batch["asa_class"] = asa
        batch["emergency"] = emergency
        batch["emergency_label"] = [int(x) for x in emergency]
        return batch

    name = split_asa_feature.__name__
    ds = map_dataset_dict_with_fingerprint(
        dataset_dict=ds,
        fingerprint_base=name if static_fingerprints else None,
        function=split_asa_feature,
        desc=name,
        batched=True,
        num_proc=num_proc,
    )

    # Remove examples with ASA class 6
    ds = ds.filter(
        lambda example: example["asa_class"] != "VI",
        desc="Remove ASA VI",
        num_proc=num_proc,
    )
    logger.info(
        f"""After Split Emergency Status & Drop ASA VI examples.
        Shape: {ds.shape} 
        Num Samples: {sum(ds.num_rows.values())}"""
    )

    def combine_IV_V(batch: dict[str, list]):
        "Combine ASA IV-V."
        asa = []
        for x in batch["asa_class"]:
            if x in ("IV", "V"):
                x = "IV-V"
            asa += [x]
        batch["asa_class"] = asa
        batch["asa_label"] = [asa_label2id[x] for x in asa]
        return batch

    name = combine_IV_V.__name__
    ds = map_dataset_dict_with_fingerprint(
        dataset_dict=ds,
        fingerprint_base=name if static_fingerprints else None,
        function=combine_IV_V,
        desc=name,
        batched=True,
        num_proc=num_proc,
    )
    logger.info(
        f"""After Combine ASA IV-V.
        Shape: {ds.shape} 
        Num Samples: {sum(ds.num_rows.values())}"""
    )
    return ds


def transform_dataset(
    ds: Union[Dataset, DatasetDict],
    task: str = None,
    tokenizer: Union[AutoTokenizer, PreTrainedTokenizerFast] = None,
    asa_label2id: dict = None,
    seq_max_length: int = 512,
    static_fingerprints: bool = False,
    num_proc: int = 4,
) -> Union[Dataset, DatasetDict]:
    "Apply data cleaning & transformations."
    # Select Variables in Dataset for a Specific Modeling Task
    ds = select_dataset_subset_for_task(ds, task=task)

    # Setup Tokenizer
    tokenizer = prepare_tokenizer(tokenizer=tokenizer, seq_max_length=seq_max_length)

    # Get "input_feature_name"
    task_to_input_feature_name = {
        "procedure-asa": "ProcedureDescription",
        "diagnosis-asa": "DiagnosisDescription",
        "hpi-asa": "HPI",
        "ros-asa": "ROS",
        "pmsh-asa": "PMSH",
        "meds-asa": "Medications",
        "note-asa": "NoteText",
        "note512-asa": "NoteText",
    }
    input_feature_name = task_to_input_feature_name[task]

    # Transform Input Feature Variable
    ds = transform_dataset_inputs(
        ds=ds,
        input_feature_name=input_feature_name,
        tokenizer=tokenizer,
        seq_max_length=seq_max_length,
        static_fingerprints=static_fingerprints,
        num_proc=num_proc,
    )

    # Transform ASA-PS Output Variable
    ds = transform_dataset_outputs(
        ds=ds,
        asa_label2id=asa_label2id,
        static_fingerprints=static_fingerprints,
        num_proc=num_proc,
    )

    # If task=note512-asa, make "TruncatedNoteText" feature & replace "input_text"
    # with text truncated to 512 tokens.  This has no effect on input_ids
    # since new feature is the text that is decoded from the input_ids.
    # Dataset caching fingerprints depend on everything that preceeds it,
    # so we place this at end of preprocessing pipeline so we don't mess with
    # previously created caches.
    if task == "note512-asa":
        ds = make_truncated_note_text(
            ds=ds,
            input_feature_name=input_feature_name,
            tokenizer=tokenizer,
            seq_max_length=seq_max_length,
            static_fingerprints=static_fingerprints,
            num_proc=num_proc,
        )

    # Format as Pytorch Tensors for Batch Dataloader
    ds.set_format(
        type="torch",
        columns=[
            "attention_mask",
            "input_ids",
            "token_type_ids",
            "asa_label",
            "emergency_label",
        ],
        output_all_columns=True,
    )
    return ds
