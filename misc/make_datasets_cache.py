# This script shows how to load lightning datamodule with different dataset variants for each task,
# and in the process generates datasets cache and fasttext compatible training format for each task.

from __future__ import annotations

import logging
import os
from pathlib import Path

import hydra
from pytorch_lightning import seed_everything
from src.modeling.datamodule import DataModule
from models.utils import resolve_paths_and_save_config

log = logging.getLogger(__name__)


# Load default hydra config for configuring datamodule
with hydra.initialize(config_path="../conf", version_base=None):
    cfg = hydra.compose(
        config_name="bioclinicalbert_config",
    )
    cfg = resolve_paths_and_save_config(cfg, file_ref=__file__)

# Reproducibility
seed_everything(seed=cfg.general.seed, workers=True)

# Load Data with dataset created at default path data/../../processed/
dataset_path = None

# Load Data w/ dataset at specific path (not in default data folder)
# dataset_path = Path.cwd() / "data" / cfg.datamodule.dataset_name

datamodule = DataModule(
    project_dir=cfg.general.project_root,
    dataset_path=dataset_path,
    seq_max_length=cfg.model.data.seq_max_length,
    **cfg.datamodule,
)

datamodule.prepare_data()

## Preprocess dataset for each specific task & export into fasttext compatible training format
# procedure-asa task
datamodule.task_name = "procedure-asa"
datamodule.has_prepare_data = False
datamodule.prepare_data()

datamodule.export_fasttext(
    split="train",
    combine=False,
    task_name=datamodule.task_name,
    input_feature_name="input_text",
    output_label_name="asa_label",
)

# diagnosis-asa task
datamodule.task_name = "diagnosis-asa"
datamodule.has_prepare_data = False
datamodule.prepare_data()

datamodule.export_fasttext(
    split="train",
    combine=False,
    task_name=datamodule.task_name,
    input_feature_name="input_text",
    output_label_name="asa_label",
)

# hpi-asa task
datamodule.task_name = "hpi-asa"
datamodule.has_prepare_data = False
datamodule.prepare_data()

datamodule.export_fasttext(
    split="train",
    combine=False,
    task_name=datamodule.task_name,
    input_feature_name="input_text",
    output_label_name="asa_label",
)

# ros-asa task
datamodule.task_name = "ros-asa"
datamodule.has_prepare_data = False
datamodule.prepare_data()

datamodule.export_fasttext(
    split="train",
    combine=False,
    task_name=datamodule.task_name,
    input_feature_name="input_text",
    output_label_name="asa_label",
)

# pmsh-asa task
datamodule.task_name = "pmsh-asa"
datamodule.has_prepare_data = False
datamodule.prepare_data()

datamodule.export_fasttext(
    split="train",
    combine=False,
    task_name=datamodule.task_name,
    input_feature_name="input_text",
    output_label_name="asa_label",
)

# meds-asa task
datamodule.task_name = "meds-asa"
datamodule.has_prepare_data = False
datamodule.prepare_data()

datamodule.export_fasttext(
    split="train",
    combine=False,
    task_name=datamodule.task_name,
    input_feature_name="input_text",
    output_label_name="asa_label",
)

# note-asa task (uses whole note)
datamodule.task_name = "note-asa"
datamodule.has_prepare_data = False
datamodule.prepare_data()

datamodule.export_fasttext(
    split="train",
    combine=False,
    task_name=datamodule.task_name,
    input_feature_name="input_text",
    output_label_name="asa_label",
)

# note512-asa task (note truncated to 512 WordPiece tokens, then decoded back to text)
datamodule.task_name = "note512-asa"
datamodule.has_prepare_data = False
datamodule.prepare_data()

datamodule.export_fasttext(
    split="train",
    combine=False,
    task_name=datamodule.task_name,
    input_feature_name="input_text",
    output_label_name="asa_label",
)

datamodule.dataset_dict["train"][0]
