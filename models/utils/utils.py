from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, OmegaConf
from src.modeling.datamodule import DataModule

log = logging.getLogger(__name__)


def prepare_datamodule(
    cfg: Union[dict, DictConfig], dataloader_batch_size: int = 32
) -> DataModule:
    cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
    # Per-Run Dataset Location
    if cfg.general.export_dataset:
        dataset_path = (
            Path(cfg.general.output_dir) / "data" / cfg.datamodule.dataset_name
        )
    else:
        dataset_path = None

    # Only BERT model requires dataloader and batch_size should be taken from cfg
    # other models just need a batch_size specified so dataloader can be initialized
    dataloader_batch_size = (
        cfg.model.params.batch_size
        if cfg.model.model_type == "bioclinicalbert"
        else dataloader_batch_size
    )

    # Load Data
    datamodule = DataModule(
        project_dir=cfg.general.project_root,
        dataset_path=dataset_path,
        parallel_preprocess_dataset=cfg.general.parallel_preprocess_dataset,
        dataloader_batch_size=dataloader_batch_size,
        seq_max_length=cfg.model.data.seq_max_length,
        **cfg.datamodule,
    )
    datamodule.prepare_data()
    log.info(f"Dataset Path: {datamodule.dataset_path}")
    return datamodule
