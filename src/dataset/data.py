from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Union
from azureml.core import Workspace
from .cases import Cases
from .notes import Notes
from .make_dataset import MakeDataset
import datasets
from datasets import DatasetDict


class Data(
    Cases,
    Notes,
    MakeDataset,
):
    """This class is entry point for data access and holds references
    to various pieces of data for dataset creation.  This class combines
    functionality from other mix-in classes.

    Methods in this class manage data download to local directory and data
    cleaning before combining data tables into a dataset fit for
    model training.

    In general, the dataframes stored as attributes in the Data object
    are indexed by a primary key unique within that dataframe:
        cases: ProcedureID
        notes: NoteID
    """

    def __init__(
        self,
        project_dir: Union[Path, str] = None,
        datastore_blob_path: Union[Path, str] = None,
        workspace_path: str = None,
        dataset_name: str = "hpi-pmsh-ros-meds-asa",
        seed: int = 42,
    ):
        """
        Args:
            project_dir (path, str): top-level dir for project on local machine
            datastore_blob_path (path, str): virtual path in datastore where raw data is stored
            workspace_path (str): path for azure workspace.  If `None`, will get default workspace.
        """
        super().__init__()
        self.seed = seed
        self.project_dir = Path(project_dir)
        self.datastore_blob_path = Path(datastore_blob_path)
        self.dataset_name = dataset_name
        # References to data on local filesystem
        self.data_dir = self.project_dir / self.datastore_blob_path.parent
        self.raw_data_dir = self.project_dir / self.datastore_blob_path
        self.interim_data_dir = self.data_dir / "interim"
        self.processed_data_dir = self.data_dir / "processed"

        # References to azure datastore
        self.workspace = Workspace.from_config(path=workspace_path)
        self.datastore = self.workspace.get_default_datastore()

        # Reference raw data tables
        self.cases = None
        self.notes = None

    def _download_data(self, overwrite: bool = False) -> None:
        "Download data from datastore."
        self.datastore.download(
            target_path=self.project_dir,
            prefix=self.datastore_blob_path,
            show_progress=True,
            overwrite=overwrite,
        )

    def maybe_download_data(self, force_download: bool = False):
        if not self.raw_data_dir.exists():
            self._download_data()
        elif force_download:
            self._download_data(overwrite=True)
        self.interim_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def make_cases(self) -> None:
        self.maybe_download_data()
        # Load Data
        cases = pd.read_parquet(self.raw_data_dir / "cases.parquet")
        # Clean Cases
        cases = self.clean_cases(cases)
        cases = self.filter_invalid_cases(
            cases, cache_path=self.interim_data_dir / "filtered_cases.parquet"
        )
        cases = self.merge_all_overlap_cases(
            cases, cache_path=self.interim_data_dir / "merged_cases.parquet"
        )
        self.cases = cases

    def get_cases(self) -> pd.DataFrame:
        self.make_cases()
        return self.cases

    def make_notes(self) -> None:
        self.maybe_download_data()
        # Load Data
        notes = pd.read_parquet(self.raw_data_dir / "notes.parquet")
        # Clean Notes, Find Preanesthesia Notes, Segment & Extract Note Fields
        notes = self.clean_notes(notes)
        notes = self.is_preanesthesia(
            notes, cache_path=self.interim_data_dir / "is_preanesthesia.parquet"
        )
        notes = self.segment_preanesthesia_notes(
            notes, cache_path=self.interim_data_dir / "preanesthesia_extracted.parquet"
        )
        # Combine PMH and PSH from segmented notes to make PMSH column
        notes = self.make_pmsh(notes)
        self.notes = notes

    def get_notes(self) -> pd.DataFrame:
        self.make_notes()
        return self.notes

    def make_data_tables(self, force_download: bool = False) -> Data:
        "Download, clean, load data."
        self.maybe_download_data(force_download)
        self.make_cases()
        self.make_notes()
        return self

    def make_dataset(self, dataset_name: str = None) -> DatasetDict:
        dataset_name = self.dataset_name if dataset_name is None else dataset_name
        # Make Cases & Notes Data Tables
        if self.cases is None or self.notes is None:
            self.make_data_tables()
        # Associate Cases & Notes Tables
        association_fn = {
            "hpi-pmsh-ros-meds-asa": self.make_hpi_pmsh_ros_meds_asa_association,
            "hpi-asa": self.make_hpi_asa_association,
            "pmsh-asa": self.make_pmsh_asa_association,
            "ros-asa": self.make_ros_asa_association,
            "meds-asa": self.make_meds_asa_association,
        }
        fn = association_fn[dataset_name]
        df = fn()
        # Join Cases & Notes into Single Dataframe
        df = self.make_dataframe(df)
        df = self.format_dataframe(df)
        # Make Train/Validation/Test splits
        dataset_dict = self.make_train_validation_test_datasets(df=df, seed=self.seed)
        return dataset_dict

    def get_dataset(self, cache_path: Union[Path, str] = None) -> DatasetDict:
        "Get dataset or create it if does not yet exist at `cache_path`."
        cache_path = (
            self.processed_data_dir / self.dataset_name
            if cache_path is None
            else Path(cache_path)
        )
        try:
            dataset_dict = datasets.load_from_disk(Path(cache_path).as_posix())
        except:  # noqa: E722
            dataset_dict = self.make_dataset()
            Path(cache_path.parent).mkdir(parents=True, exist_ok=True)
            dataset_dict.save_to_disk(cache_path)
        return dataset_dict
