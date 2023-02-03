import inspect
from pathlib import Path
from time import localtime, strftime
from typing import Union

import numpy as np
import ray
from flaml.tune.trial import flatten_dict, unflatten_dict
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def dict_to_omegaconf(d: dict, delimiter: str = "|") -> DictConfig:
    d = {k: float(v) if isinstance(v, np.float64) else v for k, v in d.items()}
    d = unflatten_dict(d, delimiter=delimiter)
    return OmegaConf.create(d)


def omegaconf_to_dict(cfg: DictConfig, delimiter: str = "|") -> dict:
    return flatten_dict(cfg, delimiter=delimiter)


def make_output_subdirs(output_dir_root: Union[Path, str]) -> Path:
    "Creates Hydra-style output directory path = output_dir_root/YYYY-MM-DD/HH-MM-SS."
    ymd_dir, hms_dir = strftime("%Y-%m-%d_%H-%M-%S", localtime()).split("_")
    return Path(output_dir_root) / ymd_dir / hms_dir


def make_output_dir(project_root: Union[Path, str]) -> Path:
    "Returns Hydra-style output directory."
    return make_output_subdirs(Path(project_root) / "outputs")


def get_output_dir(project_root: Union[Path, str]) -> Path:
    "If Hydra initialized, get Hydra output config, otherwise, make Hydra-style project_dir."
    # If HydraConfig exists, then hydra.main() used & will resolve to default hydra output path,
    # Otherwise, create similar path if not using hydra.main()
    if HydraConfig.initialized():
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    else:
        output_dir = make_output_dir(project_root)
    return output_dir


def set_project_root(
    cfg: DictConfig,
    file_ref: Union[Path, str] = None,
    project_root: Union[Path, str] = None,
) -> DictConfig:
    """If no value provided for `project_root` in `cfg`, then,
    set it to a default value based on whether hydra.main() is used or not
    to compose the `cfg`.
    """
    # If explicit project_root is passed in, use it.
    if project_root:
        project_root = Path(project_root)
    else:
        if cfg.general.project_root is None:
            # By default, set file reference to calling file
            file_ref = inspect.stack()[1].filename if file_ref is None else file_ref
            project_root = Path(file_ref).parent.parent.parent.resolve()
        else:
            project_root = Path(cfg.general.project_root)
    cfg.general.project_root = project_root.as_posix()
    return cfg


def set_output_dir(cfg: DictConfig) -> DictConfig:
    """If no value provided for `output_dir` in `cfg`, then,
    set it to a default value based on whether hydra.main() is used or not
    to compose the `cfg`.
    """
    if cfg.general.output_dir is None:
        output_dir = get_output_dir(project_root=cfg.general.project_root)
    else:
        output_dir = Path(cfg.general.output_dir)
    cfg.general.output_dir = output_dir.as_posix()
    return cfg


def set_ray_tune_checkpoint_dir(cfg: DictConfig) -> DictConfig:
    """If in a Ray-Tune session, sets path for PytorchLightning Checkpoint
    saving to be in same directory as Ray-Tune trial.  This is distinct from
    MLflow model logging.
    """
    if ray.tune.is_session_enabled():
        # Ray-Tune defaults to checkpoint dir in each trial
        cfg.general.checkpoint_dir = None
    else:
        # Default checkpoint dir for single run
        cfg.general.checkpoint_dir = (
            Path(cfg.general.output_dir) / "checkpoints"
        ).as_posix()
    return cfg


def save_config(cfg: DictConfig, save_dir: Union[Path, str] = None):
    """Saves config to outputs/.hydra_compose/config.yaml if hydra.main() is
    not used.  If hydra.main() is used, this method does nothing.
    By default hydra.main() will save the config to outputs/.hydra/config.yaml.
    This function assumes appropriate output_dir is set already."""
    if not HydraConfig.initialized():
        save_dir = Path(cfg.general.output_dir) if save_dir is None else save_dir
        save_dir = Path(cfg.general.output_dir) / ".hydra_compose"
        save_dir.mkdir(parents=True, exist_ok=True)
        f = save_dir / "config.yaml"
        if not f.exists():
            OmegaConf.save(config=cfg, f=f)


def resolve_paths_and_save_config(
    cfg: DictConfig,
    file_ref: Union[Path, str] = None,
    project_root: Union[Path, str] = None,
) -> DictConfig:
    "When using this method in a notebook, recommend manually setting `file_ref` or `project_root`."
    # By default, set file reference to calling file
    file_ref = inspect.stack()[1].filename if file_ref is None else file_ref
    cfg = set_project_root(cfg, file_ref, project_root)
    cfg = set_output_dir(cfg)
    cfg = set_ray_tune_checkpoint_dir(cfg)
    save_config(cfg)
    return cfg
