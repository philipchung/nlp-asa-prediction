import logging
from pathlib import Path
from typing import Union

import datasets
import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    clear_datasets_cache(**cfg)


def clear_datasets_cache(
    name: str = "hpi-pmsh-ros-meds-asa", 
    datasets_dir: Union[str, Path] = None,
) -> None:
    "Clears datasets cache."
    datasets_dir = Path(datasets_dir)
    ds_path = datasets_dir / name
    ds = datasets.load_from_disk(ds_path.as_posix())
    result = ds.cleanup_cache_files()
    print(result)

if __name__ == "__main__":
    main()
