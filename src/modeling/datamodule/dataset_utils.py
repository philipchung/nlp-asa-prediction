from __future__ import annotations

import hashlib
from datasets import Dataset, DatasetDict

from typing import Union


def map_dataset_dict_with_fingerprint(
    dataset_dict: DatasetDict = None, fingerprint_base: str = None, **kwargs
) -> DatasetDict:
    """Applies map to each dataset in `dataset_dict` by passing `kwargs` into map.
    This method allows specifying a static `fingerprint_base` which will be combined
    with the key name in dataset_dict to create a string for which a static hash is
    generated.  This static hash will be used to generate a cache so that each
    dataset in `dataset_dict` will have a unique static hash that is all derived from
    the `fingerprint_base` value.
    """
    if "desc" in kwargs:
        desc = kwargs.pop("desc")
    elif fingerprint_base is not None:
        desc = kwargs["fingerprint_base"]
    else:
        desc = ""

    new_dict = {}
    for split_name, dset in dataset_dict.items():
        if fingerprint_base is None:
            static_hash = None
        else:
            fingerprint = f"{fingerprint_base}-{split_name}"
            static_hash = hashlib.md5(fingerprint.encode("utf8")).hexdigest()
        # Apply map() to datasets
        new_dict[split_name] = dset.map(
            new_fingerprint=static_hash, desc=f"{split_name}:{desc}", **kwargs
        )
    return DatasetDict(new_dict)


def select_dataset_columns(
    dataset: Union[Dataset, DatasetDict], columns: Union[str, list[str]] = None
) -> Union[Dataset, DatasetDict]:
    "Select subset of columns from dataset."
    if columns is None:
        return dataset
    else:
        columns = [columns] if not isinstance(columns, list) else columns
    if isinstance(dataset, Dataset):
        dataset_col_names = dataset.column_names
    elif isinstance(dataset, DatasetDict):
        first_split = list(dataset.keys())[0]
        dataset_col_names = dataset[first_split].column_names
    else:
        raise ValueError(f"Unknown `dataset` type {dataset}, must be type Dataset or DatasetDict.")
    # Select specific columns in dataset
    columns_to_remove = sorted(list(set(dataset_col_names) - set(columns)))
    # Ensure we keep column `index`, which is original index of data item before splits generated
    columns_to_remove = [col for col in columns_to_remove if col != "index"]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def filter_dataset_columns(
    dataset: Union[Dataset, DatasetDict], columns: Union[str, list[str]] = None
) -> Union[Dataset, DatasetDict]:
    "Remove subset of columns from dataset."
    if columns is None:
        return dataset
    else:
        columns = [columns] if not isinstance(columns, list) else columns
    if isinstance(dataset, Dataset):
        dataset_col_names = dataset.column_names
    elif isinstance(dataset, DatasetDict):
        first_split = list(dataset.keys())[0]
        dataset_col_names = dataset[first_split].column_names
    else:
        raise ValueError(f"Unknown `dataset` type {dataset}, must be type Dataset or DatasetDict.")
    # Remove columns only if they appear in dataset
    columns_to_remove = sorted(list(set(columns).intersection(dataset_col_names)))
    # Ensure we keep column `index`, which is original index of data item before splits generated
    columns_to_remove = [col for col in columns_to_remove if col != "index"]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset
