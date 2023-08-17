from __future__ import annotations

from pathlib import Path
from typing import Union

import mlflow
import pandas as pd
from azureml.core import Workspace
from omegaconf import OmegaConf


def get_mlflow_runs(
    parent_run_ids: Union[dict, str, Path] = None,
    child_run_types: Union[str, list[str]] = [
        "train_runs",
        "test_runs",
        "model_runs",
        "shap_runs",
    ],
) -> dict[str, pd.DataFrame]:
    """Convenience function to evaluation run experiments, including
    train, test, model logging child runs for each model type and task.

    Args:
        parent_run_ids (Union[dict, str, Path], optional): If dict, expects a dict of dict
            of parent run ids.  If str or Path, expects a .yaml file which specifies parent
            run ids.  Top level key is model type ("rf", "svm", "fasttext", "bioclincialbert"),
            and second level key is task type ("procedure", "diagnosis", "hpi", "ros", "pmsh",
            "meds", "note", "note512").
        child_run_types: (Union[str, list[str]], optional): Can skip getting all child runs
            by specifying only child run types needed.  Valid values: "train_runs",
            "test_runs", "model_runs", "shap_runs". If "None" or empty then no child runs are returned.
            "parent_runs" are always returned.

    Returns:
        dict[str, pd.DataFrame]: keys include "parent_runs", "train_runs", "test_runs",
            "model_runs".  Each value is a pandas dataframe with corresponding runs for
            each model type and task type.
    """
    # Run IDs for Evaluation Parent Runs
    if parent_run_ids is None:
        # Load default parent run ids
        parent_run_ids = OmegaConf.load(
            Path(__file__).parent / "evaluation_run_ids.yaml"
        )
    elif isinstance(parent_run_ids, Path) or isinstance(parent_run_ids, str):
        parent_run_ids = OmegaConf.load(Path(parent_run_ids))
    elif isinstance(parent_run_ids, dict):
        # parent_run_ids is already a dict
        pass
    else:
        raise ValueError("Argument `parent_run_ids` must be type dict, str, or Path.")

    if not child_run_types:
        child_run_types = []
    elif isinstance(child_run_types, str):
        child_run_types = [child_run_types]

    ws = Workspace.from_config()
    mlflow_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)

    # Get Run Objects for each Parent Run
    parent_run_ids_df = pd.DataFrame.from_dict(parent_run_ids, orient="index")
    parent_runs_df = parent_run_ids_df.applymap(
        lambda run_id: mlflow.get_run(run_id) if run_id else None
    )

    outputs = {"parent_runs": parent_runs_df}

    if child_run_types:
        # Child Run IDs
        child_run_ids_df = parent_runs_df.applymap(get_evaluation_child_run_ids)
        # Get Run Objects for each Child Run
        if "train_runs" in child_run_types:
            train_run_ids_df = child_run_ids_df.applymap(lambda x: x["train_run_id"])
            train_runs_df = train_run_ids_df.applymap(
                lambda run_id: mlflow.get_run(run_id) if run_id else None
            )
            outputs["train_runs"] = train_runs_df
        if "test_runs" in child_run_types:
            test_run_ids_df = child_run_ids_df.applymap(lambda x: x["test_run_id"])
            test_runs_df = test_run_ids_df.applymap(
                lambda run_id: mlflow.get_run(run_id) if run_id else None
            )
            outputs["test_runs"] = test_runs_df
        if "model_runs" in child_run_types:
            model_run_ids_df = child_run_ids_df.applymap(lambda x: x["model_run_id"])
            model_runs_df = model_run_ids_df.applymap(
                lambda run_id: mlflow.get_run(run_id) if run_id else None
            )
            outputs["model_runs"] = model_runs_df
        if "shap_runs" in child_run_types:
            shap_run_ids_df = child_run_ids_df.applymap(lambda x: x["shap_run_id"])
            shap_runs_df = shap_run_ids_df.applymap(
                lambda run_id: mlflow.get_run(run_id) if run_id else None
            )
            outputs["shap_runs"] = shap_runs_df
    return outputs


def get_evaluation_child_run_ids(
    parent_run: Union[str, mlflow.entities.Run] = None
) -> dict[str, str]:
    if isinstance(parent_run, str):
        parent_run = mlflow.get_run(run_id=parent_run)
    elif isinstance(parent_run, mlflow.entities.Run):
        pass
    else:
        raise ValueError("Argument `parent_run` must be a run_id or mlflow Run.")

    query = f"tags.ParentRunID = '{parent_run.info.run_id}'"
    runs = mlflow.search_runs(
        experiment_ids=parent_run.info.experiment_id, filter_string=query
    )
    # Train Child Runs
    train_run = runs[runs["tags.RunType"] == "Train"]
    train_run_id = train_run.run_id.item() if not train_run.empty else None
    # Test Child Runs
    test_run = runs[runs["tags.RunType"] == "Test"]
    test_run_id = test_run.run_id.item() if not test_run.empty else None
    # Model Child Runs
    model_run = runs[runs["tags.RunType"] == "ModelLogging"]
    model_run_id = model_run.run_id.item() if not model_run.empty else None
    # Shap Child Runs
    shap_run = runs[runs["tags.RunType"] == "Shap"]
    shap_run_id = shap_run.run_id.item() if not shap_run.empty else None

    return {
        "train_run_id": train_run_id,
        "test_run_id": test_run_id,
        "model_run_id": model_run_id,
        "shap_run_id": shap_run_id,
    }


def get_baseline_runs() -> dict[str, mlflow.entities.Run]:
    baseline_run_ids = OmegaConf.load(Path(__file__).parent / "baseline_run_ids.yaml")

    ws = Workspace.from_config()
    mlflow_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)

    random_classifier = mlflow.get_run(run_id=baseline_run_ids.random_classifier.test)
    age_meds_classifier = mlflow.get_run(run_id=baseline_run_ids.age_meds_classifier.test)

    return {"random_classifier": random_classifier, "age_meds_classifier": age_meds_classifier}
