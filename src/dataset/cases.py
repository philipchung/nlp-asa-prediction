import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import cpu_count
from src.dataset.utils import TimeRange, merge_overlap_timeranges, parallel_process
from typing import Union


class Cases:
    """Mix-in Class containing methods for processing Cases data."""

    def __init__(self, cases: pd.DataFrame = None):
        self.cases = cases

    def clean_cases(self, cases: pd.DataFrame = None) -> pd.DataFrame:
        cases = self.cases if cases is None else cases
        # Harmonize values {"GEN", "General"} -> "General"
        cases.AnesType = cases.AnesType.apply(
            lambda x: "General" if (x == "GEN") else x
        )
        return cases

    def filter_invalid_cases(
        self, cases: pd.DataFrame = None, cache_path: Union[Path, str] = None
    ) -> pd.DataFrame:
        cases = self.cases if cases is None else cases

        try:
            cases = pd.read_parquet(path=cache_path)
        except:  # noqa: E722
            cases = self._filter_invalid_cases(cases)
            Path(cache_path.parent).mkdir(parents=True, exist_ok=True)
            cases.to_parquet(path=cache_path)
        return cases

    def _filter_invalid_cases(self, cases: pd.DataFrame) -> pd.DataFrame:
        """Removes cases with invalid time ranges.  This includes:
        1. `AnesStart` == `NaT`
        2. `AnesEnd` == `NaT`
        3. `AnesEnd` before `AnesStart`
        """
        valid1 = cases.loc[cases.AnesStart.apply(pd.notnull)]
        valid2 = cases.loc[cases.AnesEnd.apply(pd.notnull)]
        tqdm.pandas(desc="Filtering Invalid Cases")
        duration_seconds = cases.progress_apply(
            lambda row: TimeRange(
                start=row.AnesStart, end=row.AnesEnd
            ).duration.total_seconds(),
            axis=1,
        )
        positive_duration = duration_seconds >= 0
        valid3 = cases.loc[positive_duration]
        valid_indices = list(
            set(valid1.index)
            .intersection(set(valid2.index))
            .intersection(set(valid3.index))
        )
        new_cases = cases.loc[valid_indices, :].sort_index(ascending=True)
        print(
            f"Filtering Invalid Cases.\n"
            f"Cases shape: {cases.shape} --> {new_cases.shape}."
        )
        return new_cases

    def merge_all_overlap_cases(
        self, cases: pd.DataFrame = None, cache_path: Union[Path, str] = None
    ) -> pd.DataFrame:
        "Merge cases with overlapping time intervals.  Cache result."
        cases = self.cases if cases is None else cases

        try:
            cases = pd.read_parquet(path=cache_path)
        except:  # noqa: E722
            cases = self._merge_all_overlap_cases(cases)
            Path(cache_path.parent).mkdir(parents=True, exist_ok=True)
            cases.to_parquet(path=cache_path)
        return cases

    def _merge_all_overlap_cases(self, cases: pd.DataFrame) -> pd.DataFrame:
        """Given dataframe of cases with overlapping time intervals for
        multiple `PersonID`, merge all overlapping time intervals for each `PersonID`.
        Args:
            cases: pandas dataframe of cases with datetime objects in columns
                `AdmitDateTime`, `DischargeDateTime`.  This dataframe may have encounter
                intervals for multiple `PersonID`.
        Returns:
            Original `cases` dataframe, but with all rows with overlapping
            time intervals dropped and replaced with new rows with information merged
            from the overlapping time intervals.
        """
        merged_cases = parallel_process(
            iterable=cases.groupby("PersonID", sort=False),
            function=merge_overlapped_cases,
            use_args=True,
            n_jobs=cpu_count(),
        )

        new_cases = pd.concat(merged_cases, axis=0).sort_values(
            by=["PersonID", "AnesStart"], ascending=[True, True]
        )
        print(
            f"Merged Overlap Cases.\n"
            f"Cases shape: {cases.shape} --> {new_cases.shape}."
        )
        return new_cases


# Note: parallel_process requires functions to be pickleable and not embedded within classes
def merge_overlapped_cases(id, case_group):
    return merge_overlap_timeranges(
        df=case_group, start_name="AnesStart", end_name="AnesEnd"
    )
