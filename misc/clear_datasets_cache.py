import logging

import hydra
import mlflow
import datasets
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Union

log = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    clear_datasets_cache(**cfg)


def clear_datasets_cache(
    name: str = "hpi-pmsh-ros-meds-asa", 
    datasets_dir: Union[str, Path] = "/home/azureuser/cloudfiles/code/Users/chungph/presurgnlp-az/data/id/v3/processed/"
) -> None:
    "Clears datasets cache."
    datasets_dir = Path(datasets_dir)
    ds_path = datasets_dir / name
    ds = datasets.load_from_disk(ds_path.as_posix())
    result = ds.cleanup_cache_files()
    print(result)

if __name__ == "__main__":
    main()
