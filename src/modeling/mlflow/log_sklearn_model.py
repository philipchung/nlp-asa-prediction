from pathlib import Path
from typing import Union

import pandas as pd

import mlflow


def log_sklearn_model(
    model,
    artifact_path: str = "model",
    project_root: Union[str, Path] = None,
    model_type: str = None,
    registered_model_name: str = "model",
    inputs: pd.DataFrame = None,
    outputs: Union[pd.DataFrame, pd.Series] = None,
) -> mlflow.models.model.ModelInfo:
    """Log a Sklearn Model to MLFlow as an artifact.

    Args:
        model (sci-kit learn model or pipeline): The model to log
        artifact_path (str): directory name for the logged artifact
        project_root (Path, str): project root directory
        model_type (str): "svm" for Support Vector Machine Classifier, "rf" for Random Forest.
            This selects the appropriate source code path to log with the model
        registered_model_name (str): name for logged model in model registry
        inputs (pd.DataFrame): sample input data X used to infer model input schema
        outputs (pd.DataFrame or pd.Series): sample output data y used to infer model output schema
    """
    if project_root is None:
        raise ValueError("Must provide argument `project_root`.")
    project_root = Path(project_root).resolve()
    pip_requirements_path = project_root / "requirements.txt"
    if model_type == "rf":
        script_code_path = project_root / "rf"
    elif model_type == "svm":
        script_code_path = project_root / "svm"
    else:
        raise ValueError("Argument `model_type` must be either 'rf' or 'svm'")
    lib_code_path = project_root / "src"

    signature = mlflow.models.infer_signature(
        inputs[:2].to_frame(),
        outputs[:2],
    )
    input_example = pd.DataFrame(
        [
            "37 year old man with left femur fracture, scheduled for ORIF.",
            "60 year old woman undergoing routine colonoscopy.",
        ]
    )
    # Log Model in MLModel format (sklearn flavor)
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        pip_requirements=pip_requirements_path.as_posix(),
        code_paths=[script_code_path.as_posix()],
        # code_paths=[script_code_path.as_posix(), lib_code_path.as_posix()],
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
    )
    return model_info
