import pandas as pd
import csv
from pathlib import Path
from typing import Union
import warnings


def save_csv(series: pd.Series, save_path: Union[str, Path]) -> Union[str, Path]:
    "Save CSV without additional quoting.  Must make sure delimiter does not occur in data."
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        series.to_csv(
            path_or_buf=save_path,
            sep="\n",
            header=False,
            index=False,
            mode="x",
            quoting=csv.QUOTE_NONE,
        )
    except FileExistsError:
        warnings.warn(
            f"File already exists at {save_path.as_posix()}, skipping save-file."
        )
    return save_path
