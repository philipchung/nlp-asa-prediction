from __future__ import annotations

from collections import defaultdict, namedtuple
from datetime import datetime
from itertools import chain
from typing import List, Union

import numpy as np
import pandas as pd
from src.dataset.utils import find_intersecting_sets

Overlap = namedtuple(
    "Overlap",
    [
        "leading",
        "overlap",
        "trailing",
        "leading_id",
        "overlap_ids",
        "trailing_id",
    ],
)


class TimeRange(object):
    def __init__(self, start: datetime, end: datetime, **kwargs):
        self.start = start
        self.end = end
        self.duration = self.end - self.start
        self.id = kwargs["id"] if (kwargs is not None) and ("id" in kwargs) else None

    def __eq__(self, other: TimeRange) -> bool:
        if not isinstance(other, TimeRange):
            # don't attempt to compare against unrelated types
            return NotImplemented
        if (self.start == other.start) and (self.end == other.end):
            return True
        else:
            return False

    def is_overlapped(self, time_range: TimeRange):
        "Checks to see if overlap between self and `time_range`"
        if max(self.start, time_range.start) < min(self.end, time_range.end):
            return True
        else:
            return False

    def get_overlapped_range(
        self, time_range: TimeRange, overlap_only: bool = True
    ) -> Union[TimeRange, dict]:
        """Returns overlapped range.  If no overlap, returns np.nan.
        Args:
            time_range: other time range object to compare against this object
            overlap_only: whether or not to return comprehensive time ranges
                including 3 time ranges:
                    1. time range A only
                    2. time range A overlap with time range B
                    3. time range B only
                If `False`, will only return overlap (#2).
        Returns:
            If `overlap_only` is `False`, returns only overlapping time range.
            If `overlap_only` is `True`, returns a dict of time ranges
                as described above.
        """
        A = self
        B = time_range
        if not A.is_overlapped(B):
            return np.nan

        # A starts before B
        if B.start >= A.start:
            # A ends after B
            if A.end >= B.end:
                ranges = Overlap(
                    leading=TimeRange(A.start, B.start),
                    overlap=TimeRange(B.start, B.end),
                    trailing=TimeRange(B.end, A.end),
                    leading_id=A.id,
                    overlap_ids=(A.id, B.id),
                    trailing_id=A.id,
                )
                return ranges.overlap if overlap_only else ranges
            # A ends before B
            else:
                ranges = Overlap(
                    leading=TimeRange(A.start, B.start),
                    overlap=TimeRange(B.start, A.end),
                    trailing=TimeRange(A.end, B.end),
                    leading_id=A.id,
                    overlap_ids=(A.id, B.id),
                    trailing_id=B.id,
                )
                return ranges.overlap if overlap_only else ranges
        # A starts after B
        elif B.start < A.start:
            # A ends before B
            if B.end >= A.end:
                ranges = Overlap(
                    leading=TimeRange(B.start, A.start),
                    overlap=TimeRange(A.start, A.end),
                    trailing=TimeRange(A.end, B.end),
                    leading_id=B.id,
                    overlap_ids=(A.id, B.id),
                    trailing_id=B.id,
                )
                return ranges.overlap if overlap_only else ranges
            # A ends after B
            else:
                ranges = Overlap(
                    leading=TimeRange(B.start, A.start),
                    overlap=TimeRange(A.start, B.end),
                    trailing=TimeRange(B.end, A.end),
                    leading_id=B.id,
                    overlap_ids=(A.id, B.id),
                    trailing_id=A.id,
                )
                return ranges.overlap if overlap_only else ranges

    def __repr__(self):
        return f"{self.start} --> {self.end}"


def timerange_overlap_pdist(
    u: pd.Series, v: pd.Series, reset_index: bool = True, overlap_only: bool = True
) -> pd.DataFrame:
    """Accepts two sequences of TimeRange objects and constructs a pairwise distance matrix.
    Args:
        u: sequence (list, series) of TimeRange object.  TimeRange objects
            must have callable function `get_overlapped_range()`.
        v: same as `u`
        reset_index: whether to disregard series indicies and use position in
            each series in output pairwise distance matrix.  Setting this to
            `False` retains original series indices.
        overlap_only: whether to only return overlapped TimeRange or comprehensive
            dict of all TimeRange covered by `u` and `v`.
    Returns:
        Pandas dataframe that is a pairwise distance matrix between time ranges
        in `u` and `v`.
    """
    result = pd.DataFrame(np.zeros((len(u), len(v))))
    for i, item1 in enumerate(u):
        for j, item2 in enumerate(v):
            result.iloc[i, j] = item1.get_overlapped_range(
                item2, overlap_only=overlap_only
            )
    if not reset_index:
        result = result.set_index(u.index)
        result.columns = v.index
    return result


#%% [markdown]
### Finding Overlap within Same Time Range Series
#%%
def timerange_overlap_upper_triangular(
    time_ranges: pd.Series, k: int = 1, mask: bool = False
) -> pd.DataFrame:
    """Computes distance matrix for time ranges, returning upper triangular
    matrix that shows column indices that overlap with specific row indices.
    Args:
        time_ranges: pandas series of TimeRange objects
        k: int for diagonal cut-off for upper triangular
            (k=0 includes diagnoal, k=1 excludes diagonal)
        mask: whether to mark overlapped indices as `True` to generate a mask.
            If this argument is false, then a new TimeRange object denoting
            the overlapped time interval is returned instead.
    Returns:
        Pandas dataframe that is a upper triangular distance matrix that
        shows exact time range overlaps.  If input arg mask=True, then
        returns a True/False mask instead of time range overlaps.
    """
    # Get Pairwise Distance Matrix of Time Range Overlaps
    time_range_overlap = timerange_overlap_pdist(time_ranges, time_ranges)
    # Get Upper Triangular Matrix without diagonal
    # .. result is that for row i, if col j has value, this means time_range at j overlaps with i
    time_range_overlap = pd.DataFrame(np.triu(time_range_overlap, k=k)).replace(
        {0: np.nan}
    )
    if mask:
        time_range_overlap = time_range_overlap.notnull()
    return time_range_overlap


def get_overlap_indices(pdist: pd.DataFrame):
    "Get overlap indices in pairwaise distance matrix."
    indices = np.nonzero(pdist.applymap(lambda x: x if pd.notnull(x) else 0).values)
    row_ids, col_ids = [], []
    for i, j in zip(*indices):
        row_ids.append(pdist.index[i])
        col_ids.append(pdist.columns[j])
    return (row_ids, col_ids)


def find_overlapping_timeranges(time_ranges: pd.Series) -> Union[List[pd.Series], None]:
    """Identify groups of time_ranges that overlap within `time_ranges` Series.
    Args:
        time_ranges: pandas Series of TimeRange objects.
    Returns:
        List of pandas series which are a subset of `time_ranges`.
        Each series contains time ranges that overlap with one another.
        If no overlaps in `time_ranges`, then this function returns `None`.
    """
    # Sort in Ascending Order based on Start Time
    index_name = "index" if time_ranges.index.name is None else time_ranges.index.name
    df = (
        time_ranges.apply(lambda x: {"TimeRange": x, "Start": x.start, "End": x.end})
        .apply(pd.Series)
        .sort_values(by="Start", ascending=True)
        .reset_index(drop=False)
    )

    # Get Overlapped Time Ranges
    overlap_triu_matrix = timerange_overlap_upper_triangular(df.TimeRange, k=1)
    # If no Overlapped Time Ranges, return `None`
    if overlap_triu_matrix.isnull().values.all():
        return None
    else:
        # Get Indices for Overlapped Time Ranges & Transform them into Index Ranges
        row_ids, col_ids = get_overlap_indices(overlap_triu_matrix)
        # Associate Row with Multiple Overlap columns
        id_grps_dict = defaultdict(list)
        for row_id, col_id in zip(row_ids, col_ids):
            id_grps_dict[row_id].append(col_id)
        # Cluster together row idx & associated col indices as "index groups"
        id_grps_list = []
        for k, v in id_grps_dict.items():
            id_grp = set([k] + v)
            id_grps_list.append(id_grp)
        # Merge intersecting index groups
        intersecting_id_grps = find_intersecting_sets(id_grps_list)
        # Transform into ranges
        overlap_index_ranges = [range(min(x), max(x) + 1) for x in intersecting_id_grps]
        # Select Overlap Ranges as Sub-Series in Original TimeRange,
        # making sure to reset original indicies
        tr_series_subset = [
            df.loc[x, :].set_index(keys=index_name, drop=True).TimeRange
            for x in overlap_index_ranges
        ]
    return tr_series_subset


#%% [markdown]
### Use TimeRanges to find Overlaps in DataFrames & Merge Overlaps
#%%
def find_timerange_overlap_groups(
    df: pd.DataFrame, start_name: str = "Start", end_name: str = "End"
) -> Union[List[pd.DataFrame], None]:
    "Within `df` find and return overlap row groups as list of dataframes."
    # Get Time Ranges
    time_ranges = df.apply(
        lambda row: TimeRange(start=row[start_name], end=row[end_name]), axis=1
    )
    # Get List of Overlapped Time Ranges (each overlap is sub-series of `time_ranges`)
    overlap_time_ranges = find_overlapping_timeranges(time_ranges)
    if overlap_time_ranges is None:
        return None
    else:
        # List of Dataframes, where each with cluster of overlapping time ranges
        overlap_groups = [df.loc[x.index, :] for x in overlap_time_ranges]
        return overlap_groups


def merge_overlap_timerange_group(
    df: pd.DataFrame, start_name: str = "Start", end_name: str = "End"
) -> pd.Series:
    "Merges a single overlap row group."
    # If pandas series passed in, convert to dataframe
    df = df.to_frame().T if isinstance(df, pd.Series) else df
    # Get Time Ranges into DataFrame
    time_ranges = (
        df.apply(
            lambda row: TimeRange(start=row[start_name], end=row[end_name]),
            axis=1,
        )
        .apply(lambda x: {"Start": x.start, "End": x.end, "Duration": x.duration})
        .apply(pd.Series)
        .sort_values(by=["Start", "Duration"], ascending=[True, True])
    )
    # Construct New Series with Info Merged From Overlap Group
    # Use time range with longest duration as reference
    ref_interval = df.loc[time_ranges.Duration == time_ranges.Duration.max()]
    # If 2 time ranges have same longest duration, choose first one
    ref_interval = ref_interval.iloc[0:1, :]
    new_interval = ref_interval.squeeze().copy()
    # Make New Series span entire interval of overlaps
    new_interval[start_name] = time_ranges.Start.min()
    new_interval[end_name] = time_ranges.End.max()
    return new_interval


def merge_overlap_timeranges(
    df: pd.DataFrame,
    start_name: str = "Start",
    end_name: str = "End",
) -> pd.DataFrame:
    """Given dataframe with time intervals, finds all overlapping time intervals
    and merges them.
    Args:
        df: pandas dataframe with columns given by arg `start_name` and `end_name`
            holding starting and ending values of an interval, respectively.
        start_name: str, name of column in `df` holding start values for TimeRange
        end_name: str, name of column in `df` holding end values for TimeRange
    Returns:
        Pandas dataframe `df` after all overlapping time intervals merged.
    """
    # Get overlapping TimeRange row groups as List of Dataframes
    tr_overlap_grps = find_timerange_overlap_groups(
        df=df, start_name=start_name, end_name=end_name
    )
    # If no overlap, go to next TimeRange row group
    if tr_overlap_grps is None:
        return df
    # If at least one overlap, merge each of the overlap groups detected
    else:
        merged = [
            merge_overlap_timerange_group(
                df=x, start_name=start_name, end_name=end_name
            )
            for x in tr_overlap_grps
        ]
        # Drop overlapped rows, then append new merged rows
        row_indices_to_drop = chain.from_iterable([x.index for x in tr_overlap_grps])
        new_df = df.drop(index=row_indices_to_drop).append(merged)
        return new_df
