from __future__ import annotations

import copy
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import Dataset, DatasetDict
from src.dataset.data import Data
from src.modeling.utils import compute_class_weights, save_csv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .dataset_utils import select_dataset_columns
from .preprocess import preprocess_dataset

logger = logging.getLogger("dataset_logger")


@dataclass
class DataModule(pl.LightningDataModule):
    """
    Data Module for specifying data splits and batch data loading for model training.

    This data module utilizes the custom `Data` class to download raw data,
    initial data cleaning, and transformation into a dataset.

    This data module is backed by `datasets` library, which uses apache arrow for
    memory-mapped dataloading and also caches transformations.  Caches are identified by
    unique function fingerprint hash, so these transformations are created outside of
    the DataModule class since methods within a class object will have a new fingerprint
    hash each time a new DataModule object is instantiated.

    For deep learning model training, pytorch batch data loaders are created with
    the LightningDataModule API `train_dataloader()`, `val_dataloader()`,
    `test_dataloader()`, `predict_dataloader()`.

    For classical machine learning training and scenarios where the entire dataset
    can be loaded in-memory, use `train_dataframe()`, `val_dataframe()`, `test_dataframe()`,
    `predict_dataframe().
    """

    project_dir: Union[str, Path] = Path.cwd()
    data_dir: Union[str, Path] = None
    data_blob_prefix: Union[str, Path] = Path("data/id/v3/raw/")
    dataset_path: Union[str, Path] = None
    dataset_name: str = "hpi-pmsh-ros-meds-asa"
    task_name: str = "hpi-asa"
    asa_label2id: tuple = tuple(
        {
            "I": 0,
            "II": 1,
            "III": 2,
            "IV-V": 3,
        }.items()
    )
    asa_class_weights: Union[list[int], torch.Tensor] = torch.Tensor([1, 1, 1, 1])
    asa_num_classes: int = None
    asa_class_names: tuple = None
    emergency_label2id: tuple = tuple({"Not Emergency": 0, "Emergency": 1}.items())
    emergency_class_weights: Union[list[int], torch.Tensor] = torch.Tensor([1, 1])
    emergency_num_classes: int = None
    emergency_class_names: tuple = None
    dataloader_num_workers: int = 3
    dataloader_batch_size: int = 32
    seq_max_length: int = 512
    tokenizer: Union[
        str, AutoTokenizer, PreTrainedTokenizerFast
    ] = "emilyalsentzer/Bio_ClinicalBERT"
    seed: int = 1
    parallel_preprocess_dataset: bool = False
    dataset_dict: DatasetDict = None
    train: Dataset = None
    validation: Dataset = None
    test: Dataset = None
    predict: Dataset = None
    prepare_data_per_node: bool = False
    _log_hyperparams: bool = True
    has_prepare_data: bool = False
    has_setup_fit: bool = False

    def __post_init__(self):
        # Setup Data Paths
        self.project_dir = Path(self.project_dir)
        # Path to main Data directory (which contains `raw`, `interim`, `processed` subdirs)
        if self.data_dir is None:
            self.data_dir = self.project_dir / self.data_blob_prefix.parent
        # Set default dataset_path if none specified.  This is where the dataset will be created and stored.
        # by default, it is stored in the `processed` subdir in the main Data directory.
        if self.dataset_path is None:
            self.dataset_path = self.data_dir / "processed" / self.dataset_name
        # Setup tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)
        elif isinstance(self.tokenizer, AutoTokenizer) or isinstance(
            self.tokenizer, PreTrainedTokenizerFast
        ):
            pass
        else:
            raise ValueError("Unknown data type for argument `tokenizer`.")
        # Set Tokenizer Environment Variables
        # https://stackoverflow.com/a/72926996
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Make label2id & id2label into a dict
        self.asa_label2id = dict(self.asa_label2id)
        self.asa_id2label = {v: k for k, v in self.asa_label2id.items()}
        self.emergency_label2id = dict(self.emergency_label2id)
        self.emergency_id2label = {v: k for k, v in self.emergency_label2id.items()}
        # Count num classes
        self.asa_num_classes = len(self.asa_label2id)
        self.asa_class_names = tuple(self.asa_label2id.keys())
        self.emergency_num_classes = len(self.emergency_label2id)
        self.emergency_class_names = tuple(self.emergency_label2id.keys())
        # Set class weights if provided, otherwise, will be computed when prepare_data() called
        if isinstance(self.asa_class_weights, list):
            self.asa_class_weights = torch.tensor(self.asa_class_weights)
        if isinstance(self.emergency_class_weights, list):
            self.emergency_class_weights = torch.tensor(self.emergency_class_weights)

    def prepare_data(self) -> None:
        "Data preparation.  Called only once from main process."
        if not self.has_prepare_data:
            # Download Raw Data, Transform raw data tables to Dataset
            data = Data(
                project_dir=self.project_dir,
                datastore_blob_path=self.data_blob_prefix,
                dataset_name=self.dataset_name,
                seed=self.seed,
            )

            # Create a Dataset Dict
            self.dataset_dict = data.get_dataset(cache_path=self.dataset_path)

            logger.info(f"Preparing dataset for task: {self.task_name}.")

            # Preprocess Dataset & Load Columns for Specific Prediction Task
            # Note: disable parallel because of caching issue
            # https://github.com/huggingface/datasets/issues/3172
            self.dataset_dict = preprocess_dataset(
                dataset_path=self.dataset_path,
                task_name=self.task_name,
                tokenizer=self.tokenizer,
                seq_max_length=self.seq_max_length,
                asa_label2id=self.asa_label2id,
                parallel=self.parallel_preprocess_dataset,
            )
            self.train = self.dataset_dict["train"]
            self.validation = self.dataset_dict["validation"]
            self.test = self.dataset_dict["test"]

            # Compute class weights
            self.compute_class_weights()

            # Mark Prepare Data Completed
            self.has_prepare_data = True

    def setup(self, stage: Optional[str] = None):
        "Setup called from every process in distributed setting."
        pass

    def compute_class_weights(self) -> None:
        "Compute class weights from training dataset if no manual weights set on init."
        train_dataset_df = self.train.to_pandas().set_index(keys="index")
        if self.asa_class_weights is None:
            self.asa_class_weights = torch.tensor(
                compute_class_weights(train_dataset_df, "asa_label"), dtype=torch.float
            )
        if self.emergency_class_weights is None:
            self.emergency_class_weights = torch.tensor(
                compute_class_weights(train_dataset_df, "emergency_label"),
                dtype=torch.float,
            )

    def delete_dataset(self, dataset_path: Union[str, Path] = None) -> None:
        "Deletes directory at `dataset_path`."
        dataset_path = self.dataset_path if dataset_path is None else Path(dataset_path)

        # Remove all references to accessing dataset
        self.dataset_dict = None
        self.train = None
        self.validation = None
        self.predict = None
        self.has_prepare_data = False
        self.has_setup_fit = False

        # Delete dataset directory
        shutil.rmtree(dataset_path, ignore_errors=True)
        # If parent data directory empty, remove it too
        if not any(dataset_path.parent.iterdir()):
            shutil.rmtree(dataset_path.parent, ignore_errors=True)

    def uniquify_cases_for_patient(
        self, split: Union[str, list] = "train", random_state: int = None
    ) -> None:
        """
        Create 1:1 patient:case matching by randomly choosing a single case for each patient.
        Resultant dataset will override dataset split stored in datamodule properties
        `train`, `validation`, `test`, `predict`.

        Args:
            split: dataset split.  Must be string or list with values
                {"train", "validation", "test", "predict}.
            random_state: seed used for randomly sampling case for each patient
        """
        seed = self.seed if random_state is None else random_state
        for splt in self.validate_split_arg(split):
            ds = self.dataset_dict[splt]
            ds = copy.deepcopy(ds)
            # Sample 1 surgery case for each patient
            indices = (
                ds.to_pandas()
                .set_index(keys="index")
                .groupby("PersonID")["ProcedureID"]
                .sample(n=1, random_state=seed)
                .index
            )
            # Generate new subset of dataset with 1:1 patient:case matching
            new_ds = ds.filter(lambda example: example["index"] in indices)
            setattr(self, splt, new_ds)

    def reset_split(self, split: Union[str, list[str]] = "train") -> None:
        "Reset dataset split."
        for splt in self.validate_split_arg(split):
            ds = self.dataset_dict[splt]
            ds = copy.deepcopy(ds)
            setattr(self, splt, ds)

    def validate_split_arg(self, split: Union[str, list[str]]) -> list[str]:
        "Input validation for split argument."
        split = [split] if not isinstance(split, list) else split
        output = []
        for splt in split:
            splt = splt.lower()
            if splt in ("train", "validation", "test", "predict"):
                output += [splt]
            else:
                raise ValueError(f"Unknown value {splt} for argument `split`.")
        # Enforce sort order
        sort_order = {"train": 1, "validation": 2, "test": 3, "predict": 4}
        output.sort(key=lambda x: sort_order[x])
        return output

    def train_dataframe(self, columns: Union[str, list[str]] = None) -> pd.DataFrame:
        ds = select_dataset_columns(dataset=self.train, columns=columns)
        return ds.to_pandas().set_index(keys="index")

    def val_dataframe(self, columns: Union[str, list[str]] = None) -> pd.DataFrame:
        ds = select_dataset_columns(dataset=self.validation, columns=columns)
        return ds.to_pandas().set_index(keys="index")

    def test_dataframe(self, columns: Union[str, list[str]] = None) -> pd.DataFrame:
        ds = select_dataset_columns(dataset=self.test, columns=columns)
        return ds.to_pandas().set_index(keys="index")

    def predict_dataframe(self):
        ds = select_dataset_columns(dataset=self.predict, columns=columns)
        return ds.to_pandas().set_index(keys="index")

    def train_dataloader(
        self,
        batch_size: int = None,
        shuffle: bool = False,
        num_workers: int = None,
        columns: Union[str, list[str]] = None,
    ) -> DataLoader:
        ds = select_dataset_columns(dataset=self.train, columns=columns)
        return DataLoader(
            dataset=ds,
            batch_size=batch_size if batch_size else self.dataloader_batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers if num_workers else self.dataloader_num_workers,
        )

    def val_dataloader(
        self,
        batch_size: int = None,
        shuffle: bool = False,
        num_workers: int = None,
        columns: Union[str, list[str]] = None,
    ) -> DataLoader:
        ds = select_dataset_columns(dataset=self.validation, columns=columns)
        return DataLoader(
            dataset=ds,
            batch_size=batch_size if batch_size else self.dataloader_batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers if num_workers else self.dataloader_num_workers,
        )

    def test_dataloader(
        self,
        batch_size: int = None,
        shuffle: bool = False,
        num_workers: int = None,
        columns: Union[str, list[str]] = None,
    ) -> DataLoader:
        ds = select_dataset_columns(dataset=self.test, columns=columns)
        return DataLoader(
            dataset=ds,
            batch_size=batch_size if batch_size else self.dataloader_batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers if num_workers else self.dataloader_num_workers,
        )

    def predict_dataloader(
        self,
        batch_size: int = None,
        shuffle: bool = False,
        num_workers: int = None,
        columns: Union[str, list[str]] = None,
    ) -> DataLoader:
        ds = select_dataset_columns(dataset=self.predict, columns=columns)
        return DataLoader(
            dataset=ds,
            batch_size=batch_size if batch_size else self.dataloader_batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers if num_workers else self.dataloader_num_workers,
        )

    def get_data_getter(self, split: str, getter_type: str) -> callable:
        """
        Dynamically fetch data getter.

        Args:
            split (str): data split {"train", "validation", "test", "predict"}
            getter_type (str): type of data object {"dataframe", "dataloader"}

        Returns:
            Function on this datamodule that can be invoked to obtain the
            dataframe or dataloader for specified data split.
        """
        split_name_map = {
            "train": "train",
            "validation": "val",
            "test": "test",
            "predict": "predict",
        }
        return getattr(self, f"{split_name_map[split]}_{getter_type}")

    def export_fasttext(
        self,
        split: Union[str, list[str]] = None,
        combine: bool = False,
        task_name: str = None,
        input_feature_name: str = "input_text",
        output_label_name: Union[str, list[str]] = None,
        save_dir: Union[str, Path] = None,
    ) -> Path:
        """
        Export feature as new-line separated file for fastText model training.

        By default, outputs files in fasttext format for unsupervised training.
        If `output_label_name` is specified, will output file in fasttext format for
            supervised label classification training.

        Args:
            split (str, List[str]): dataset split.  Must be string or list with values
                {"train", "validation", "test", "predict}.
            combine (bool): whether to combine the splits and export as a single file
            task_name (str): name of the task for which this exported dataset will be
                used for.  In the datasets preprocessing pipeline, "task_name"
                determines which columns are selected for "input_text".
            input_feature_name (str): name of the input text feature.  By default
                this will be "input_text".  The specific column used for "input_text"
                is determined by "task_name".
            output_label_name (str, List[str]): names of the label(s) to predict.
                If the value of this is `None` (default), then exported file will
                not contain labels and is suitable for unsupervised training.  If a
                value is specified in this argument, then exported file will contain
                labels and is suitable for supervised label classification training.
            save_dir (str, Path): directory to save exported files.  if `None`, then
                will save exported files under "fasttext" subdirectory under
                `dataset_path` location of the DataModule.  If file exists at this path,
                then will use existing file rather than export and override old file.

        Returns:
            Filepath(s) to which data is exported.
        """
        split = self.validate_split_arg(split)
        task_name = self.task_name if task_name is None else task_name
        is_classification_format = True if output_label_name else False
        train_type = "supervised" if is_classification_format else "unsupervised"
        save_dir = self.dataset_path / "fasttext" if save_dir is None else save_dir

        # Specify DataFrames to get for each split
        data_getters = []
        for splt in split:
            data_getter_fn = self.get_data_getter(split=splt, getter_type="dataframe")
            data_getters += [data_getter_fn]

        def format_supervised(row: pd.Series):
            "Convert features from multiple rows in dataframe into series of strings."
            input_text = row[input_feature_name]
            if isinstance(output_label_name, list):
                label_str = r" ".join(
                    [rf"__label__{row[x]}" for x in output_label_name]
                )
                return rf"{label_str} {input_text}"
            else:
                label_str = rf"__label__{row[output_label_name]}"
                return rf"{label_str} {input_text}"

        # Materialize Dataframes & Export to File
        labels_name = (
            ".".join(output_label_name)
            if isinstance(output_label_name, list)
            else output_label_name
        )
        if combine:
            # Export feature in all splits into a single file
            split_name = "-".join(split)
            if is_classification_format:
                df = pd.concat([fn() for fn in data_getters])
                series = df.apply(format_supervised, axis=1)
                filename = f"{train_type}.{split_name}.{task_name}.{input_feature_name}.{labels_name}"
            else:
                series = pd.concat([fn()[input_feature_name] for fn in data_getters])
                filename = f"{train_type}.{split_name}.{task_name}.{input_feature_name}"

            save_path = save_csv(series=series, save_path=save_dir / filename)
            logger.info(f"Exported: {save_path}")
            return save_path
        else:
            # Export feature from each split into a separate file
            save_paths = {}
            for splt, fn in zip(split, data_getters):
                split_name = splt
                if is_classification_format:
                    df = fn()
                    series = df.apply(format_supervised, axis=1)
                    filename = f"{train_type}.{split_name}.{task_name}.{input_feature_name}.{labels_name}"
                else:
                    series = fn()[input_feature_name]
                    filename = (
                        f"{train_type}.{split_name}.{task_name}.{input_feature_name}"
                    )
                save_path = save_csv(series=series, save_path=save_dir / filename)
                save_paths = {**save_paths, splt: save_path}
                logger.info(f"Exported: {save_paths}")
            return save_paths
