import logging

import hydra
import mlflow
from azureml.core import Workspace
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    delete_models(**cfg)


def delete_models(name: str, version_start: int = None, version_end: int = None):
    "Deletes model(s) from model registry from version start to end (inclusive)."
    version_start = version_start if version_start else 1

    # Configure MLflow
    ws = Workspace.from_config()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # Delete versions of the model
    client = MlflowClient()
    versions = list(range(version_start, version_end + 1))
    for version in versions:
        try:
            client.delete_model_version(name=name, version=version)
        except:
            log.info(f"Model: {name} Version: {version} does not exist in registry.")


if __name__ == "__main__":
    main()
