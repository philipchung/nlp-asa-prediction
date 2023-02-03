from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd
from src.modeling.datamodule import DataModule

import mlflow

from . import fasttext_flavor


def log_mlflow_model(
    model,
    model_type: str = None,
    artifact_path: str = "model",
    project_root: Union[str, Path] = None,
    code_paths: list[Union[str, Path]] = None,
    pip_requirements_path: Union[str, Path] = None,
    registered_model_name: str = "model",
    datamodule: DataModule = None,
    input_name: str = None,
    output_name: str = None,
    input_example: Union[str, pd.DataFrame] = "default",
) -> mlflow.models.model.ModelInfo:
    """Log a Sklearn Model to MLFlow as an artifact.

    Args:
        model (sklearn model or pipeline, pytorch nn.Module, fasttext model): The model to log
        model_type (str): "svm" for Support Vector Machine Classifier, "rf" for Random Forest,
            "fasttext" for FastText, "pytorch" for Pytorch.
        artifact_path (str): directory name for the logged artifact
        project_root (Path, str): project root directory
        code_paths (list[Union[Path, str]]): files/folders to include in artifact.  To include
            no files/folders, set this to value `[]`.  If set to `None`, will include default
            files/folders.
        pip_requirements_path (Path, str): path to pip requirements.txt.  If `None`, will include
            default pip requirements.txt.
        registered_model_name (str): name for logged model in model registry
        datamodule (DataModule): datamodule for which `input_name` and `output_name` will specify
            sample data used to infer model input/output schema for MLModel format.  Assumes
            datamodule has a "train_dataframe" method which returns a dataframe of training data
            that has columns corresponding to `input_name` and `output_name`
        input_name (str): input feature name
        output_name (str): output label name
            MLModel.  Used as a hint of what data to feed the model.  If not provided, no
            MLModel.  Used as a hint of what data to feed the model.  If not provided, no
            input example is used.  If string value "default" passed, the default example
            will be used.
    """
    # Get Project Root
    if project_root is None:
        if datamodule is not None:
            project_root = datamodule.project_dir.resolve()
        else:
            raise ValueError("Must provide argument `project_root`.")
    else:
        project_root = Path(project_root).resolve()

    # Get CodePaths
    if model_type not in ("rf", "svm", "fasttext", "bert"):
        raise ValueError(
            "Argument `model_type` must be 'rf', 'svm', 'fasttext', or 'bert'."
        )
    if code_paths is None:
        script_code_path = project_root / model_type
        lib_code_path = project_root / "src"
        code_paths = [script_code_path, lib_code_path]
    code_paths = [Path(x).as_posix() for x in code_paths]
    if pip_requirements_path is None:
        pip_requirements_path = (project_root / "requirements.txt").as_posix()

    # MLModel Signature
    train_df = datamodule.train_dataframe(columns=[input_name, output_name])
    input_data_sample = train_df[input_name].iloc[:5].to_frame()
    output_data_sample = train_df[output_name].iloc[:5]
    signature = mlflow.models.infer_signature(input_data_sample, output_data_sample)

    # Input Example
    if input_example == "default":
        input_example = pd.DataFrame(
            [
                "37 year old man with left femur fracture, scheduled for ORIF.",
                "60 year old woman undergoing routine colonoscopy.",
            ]
        )

    # Log Model in MLModel format
    if model_type in ("rf", "svm"):
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            pip_requirements=pip_requirements_path,
            code_paths=code_paths,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )
    elif model_type == "fasttext":
        model_info = fasttext_flavor.log_model(
            ft_model=model,
            artifact_path=artifact_path,
            pip_requirements=pip_requirements_path,
            code_paths=code_paths,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )
    elif model_type == "bert":
        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=artifact_path,
            pip_requirements=pip_requirements_path,
            code_paths=code_paths,
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
        )
    else:
        raise ValueError(
            "Argument `model_type` must be 'rf', 'svm', 'fasttext', or 'bert'."
        )

    return model_info
