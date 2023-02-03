# Based on https://github.com/harupy/fasttext-flavor/blob/master/fasttext_flavor.py
# And https://www.mlflow.org/docs/latest/_modules/mlflow/xgboost.html

from __future__ import annotations

import logging
import os
import sys

import fasttext
from importlib.metadata import version
import yaml
from src.modeling.fasttext import parse_fasttext_prediction

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "fasttext"
SERIALIZED_MODEL_FILE = "model.fasttext"

_logger = logging.getLogger(__name__)


def _get_installed_fasttext_version():
    # fasttext does not have a `__version__` attribute
    try:
        ver = version("fasttext")
    except Exception:
        try:
            ver = version("fasttext_wheel")
        except Exception:
            raise Exception
    return ver


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("fasttext")]


def get_default_conda_env():
    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["fasttext=={}".format(_get_installed_fasttext_version())],
        additional_conda_channels=None,
    )


def save_model(
    ft_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    import fasttext

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    model_data_subpath = SERIALIZED_MODEL_FILE
    model_data_path = os.path.join(path, model_data_subpath)

    # Save a FastText Model
    ft_model.save_model(model_data_path)
    ft_model_class = _get_fully_qualified_class_name(ft_model)
    pyfunc.add_to_model(
        mlflow_model,
        loader_module=__name__,
        data=model_data_subpath,
        env=_CONDA_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        fasttext_version=_get_installed_fasttext_version(),
        data=model_data_subpath,
        mode_class=ft_model_class,
        code=code_dir_subpath,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


def log_model(
    ft_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        registered_model_name=registered_model_name,
        ft_model=ft_model,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def _load_model(path):
    import fasttext

    return fasttext.load_model(path)


def _load_pyfunc(path):
    return _FastTextModelWrapper(_load_model(path))


def load_model(model_uri):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path, flavor_name=FLAVOR_NAME
    )
    model_file_path = os.path.join(
        local_model_path, flavor_conf.get("data", SERIALIZED_MODEL_FILE)
    )
    return _load_model(path=model_file_path)


class _FastTextModelWrapper:
    def __init__(self, ft_model: fasttext.FastText._FastText):
        self.ft_model = ft_model

    def predict(
        self,
        model_input: str,
        num_output_classes: int = None,
    ):
        """Make prediction from FastText model.
        Args:
            model_input (str): the input data to fit into the model.
            num_output_clases (int): Number of output classes.  If `None`, only returns
                class with highest probability.  If number of classes provided, returns
                class with highest probability and also gives probability for each class.
        Returns:
            [type]: the loaded model artifact.
        """
        if num_output_classes is None:
            return self.ft_model.predict(model_input)
        else:
            return parse_fasttext_prediction(
                self.ft_model.predict(model_input, k=num_output_classes)
            )
