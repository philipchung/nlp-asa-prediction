from __future__ import annotations

import re
from typing import Union

import mlflow
import torch
from azureml.core import Workspace
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from omegaconf import DictConfig


def get_best_child_run(
    cfg: Union[dict, DictConfig] = None, workspace: Workspace = None
) -> "tuple[mlflow.entities.Run, dict]":
    """Get Best Child Run & Parameters for Best Child Run.

    Args:
        cfg (DictConfig, dict): hydra configuration
        use_best_child_run_params (bool): if True, finds best child run from
            a hyperparameter tuning experiment or parent run.  Will search for
            child runs under `cfg.evaluate.experiment_name` if there is only one
            parent tuning run in the experiment.  If there are multiple parent tuning runs
            in the experiment, then `cfg.evaluate.parent_run_id` must be specified, and
            only child runs under that parent run will be searched.  The model parameters
            for the best child run will be used to train the model instead of the model
            parameters specified in hydra configuration.  If False, then model will be
            trained using parameters specified in hydra configuration.

    Returns:
        tuple( best child run object, best child run param dict )
    """
    # Get MLFlow Experiment used for Hyperparameter Tuning
    ws = workspace if workspace else Workspace.from_config()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    if cfg.evaluate.parent_run_id:
        parent_run = mlflow.get_run(cfg.evaluate.parent_run_id)
        experiment_id = parent_run.info.experiment_id
    elif cfg.evaluate.experiment_name:
        experiment = mlflow.get_experiment_by_name(cfg.evaluate.experiment_name)
        experiment_id = experiment.experiment_id
        parent_run = None
    else:
        raise ValueError(
            "Must specify `cfg.evaluate.parent_run_id` or `cfg.evaluate.experiment_name`.  "
            "If more than one parent run in experiment, then must specify "
            "`cfg.evaluate.parent_run_id` in argument `cfg`."
        )

    # Get Runs by Experiment ID
    runs_df = mlflow.search_runs(experiment_ids=experiment_id)

    # Filter to all runs affiliated with a specific Parent Run (both Parent and Child Runs)
    if cfg.evaluate.parent_run_id:
        runs_df = runs_df[runs_df["tags.RootRunID"] == cfg.evaluate.parent_run_id]

    # Split runs to Parent and Child Runs
    child_runs_df = runs_df[runs_df["tags.ParentChild"] == "Child"]
    parent_runs_df = runs_df[runs_df["tags.ParentChild"] == "Parent"]

    # If more than 1 parent run in experiment, must specify parent_run_id
    if parent_runs_df.shape[0] > 1:
        if not cfg.evaluate.parent_run_id:
            raise ValueError(
                "More than one parent_run in experiment, so must specify `cfg.evaluate.parent_run_id`."
            )

    # Get the Parent Run if we didn't get it already
    parent_run = (
        parent_run if parent_run else mlflow.get_run(parent_runs_df.run_id.item())
    )
    parent_run_tags = parent_run.data.tags

    # Get Best Child Run
    best_child_run_id = parent_run_tags["BestChildMLFlowRunID"]
    best_child_run = mlflow.get_run(best_child_run_id)

    # Get Parameters for Best Child Run
    best_child_run_df = child_runs_df[child_runs_df.run_id == best_child_run_id]
    param_cols = [col for col in best_child_run_df.columns if "param" in col]
    best_child_run_params = best_child_run_df.squeeze().loc[param_cols].to_dict()
    best_child_run_params = {
        k.split(".")[1]: v for k, v in best_child_run_params.items()
    }
    best_child_run_params = coerce_types(best_child_run_params)
    return best_child_run, best_child_run_params


def coerce_types(d: dict) -> dict:
    "Coerce mlflow string tag values back to original types."
    return {k: coerce(k, v) for k, v in d.items()}


def coerce(key, value):
    "Convert mlflow string output back to original types.  Works for int, float, 1D tensors, string."
    if value in ("None", "none", "Null", "null"):
        return None
    # Pattern match ints, but not floats.  Find all ints.
    int_pattern = r"(?<![\d.])([+-]?[0-9]+)(?![\d.])"
    int_matches = re.findall(int_pattern, value)

    # Pattern match floats, but not ints.  Find all floats.
    float_pattern = r"[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?"
    float_matches = re.findall(float_pattern, value)

    if len(int_matches) > 1 and len(float_matches) > 1:
        raise ValueError(
            f"Cannot coerce type for {key}:{value}.  Coercion logic recognizes {value} as both int and float."
        )

    # Treat matches in string as float first
    if len(float_matches) == 1:
        # Exactly one float
        return float(float_matches[0])
    elif len(float_matches) > 1:
        # Multiple floats.  Assume it is a tensor.
        return torch.tensor([float(x) for x in float_matches])

    # If cannot find floats, then treat matches in string as int
    if len(int_matches) == 1:
        # Exactly one integer
        return int(int_matches[0])
    elif len(int_matches) > 1:
        # Multiple integers.  Assume it is a tensor.
        return torch.tensor([int(x) for x in int_matches])

    # Check if value is Boolean
    if value == "True":
        return True
    if value == "False":
        return False

    # No pattern match.  Assume value should be string.
    return value


def set_default_mlflow_tags(tags: dict[str, str] = None) -> dict[str, str]:
    """
    Set Tags for Active Run.  Will determine if Parent or Child Run.
    Assumes Runs are not nested more than 1 level.

    Args:
        tags (dict): optional, additional tags to log to MLflow for active run.
            Tags specified here will override auto-generated default
            tags if they have the same key name.
    Returns:
        Dict of tags logged to MLflow.
    """
    run = mlflow.active_run()
    # Determine if Current Run is Parent or Child Run
    root_run_id = run.data.tags["mlflow.rootRunId"]
    run_id = run.info.run_id
    run_name = run.data.tags["mlflow.runName"]
    run_is_parent = run_id == root_run_id
    # Experiment Info
    experiment_id = run.info.experiment_id
    experiment = mlflow.get_experiment(experiment_id)
    experiment_name = experiment.name

    # Common Tags
    common_tags = {
        "RootRunID": root_run_id,
        "RunID": run_id,
        "RunName": run_name,
        "ExperimentID": experiment_id,
        "ExperimentName": experiment_name,
    }
    if run_is_parent:
        # Parent Run Tags
        tags_to_log = {
            **common_tags,
            "ParentChild": "Parent",
            "RunType": "Container",
        }
        if tags:
            for k, v in tags.items():
                tags_to_log[k] = v
    else:
        # Current Run is Child, add Parent Run Info to Child Run Tags
        parent_run = mlflow.get_run(root_run_id)
        parent_run_id = parent_run.info.run_id
        parent_run_name = parent_run.data.tags["mlflow.runName"]
        tags_to_log = {
            **common_tags,
            "ParentChild": "Child",
            MLFLOW_PARENT_RUN_ID: parent_run_id,  # required to next the child run
            "ParentRunID": parent_run_id,
            "ParentRunName": parent_run_name,
        }
        if tags:
            for k, v in tags.items():
                tags_to_log[k] = v

    # Set Tags on Current Run
    mlflow.set_tags(tags_to_log)
    return tags_to_log
