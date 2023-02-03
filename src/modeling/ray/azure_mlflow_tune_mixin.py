import logging
from typing import Callable, Dict

import mlflow
from mlflow.tracking import MlflowClient
import ray
from azureml.core import Workspace
from ray.tune.trainable import Trainable

logger = logging.getLogger(__name__)


def azure_mlflow_tune_mixin(func: Callable):
    """azure_mlflow_mixin.  A modified ray.tune.integration.mlflow.mlflow_mixin
    which supports getting mlflow context and credentials from AzureML service.
    This mixin always grabs the default mlflow tracking URI from AzureML workspace.
    `tracking_uri` argument in `config` is ignored.

    MLflow (https://mlflow.org) Tracking is an open source library for
    recording and querying experiments. This Ray Tune Trainable mixin helps
    initialize the MLflow API for use with the ``Trainable`` class or the
    ``@mlflow_mixin`` function API. This mixin automatically configures MLflow
    and creates a run in the same process as each Tune trial. You can then
    use the mlflow API inside the your training function and it will
    automatically get reported to the correct run.

    For basic usage, just prepend your training function with the
    ``@mlflow_mixin`` decorator:

    .. code-block:: python

        from ray.tune.integration.mlflow import mlflow_mixin

        @mlflow_mixin
        def train_fn(config):
            ...
            mlflow.log_metric(...)

    You can also use MlFlow's autologging feature if using a training
    framework like Pytorch Lightning, XGBoost, etc. More information can be
    found here
    (https://mlflow.org/docs/latest/tracking.html#automatic-logging).

    .. code-block:: python

        from ray.tune.integration.mlflow import mlflow_mixin

        @mlflow_mixin
        def train_fn(config):
            mlflow.autolog()
            xgboost_results = xgb.train(config, ...)

    The MlFlow configuration is done by passing a ``mlflow`` key to
    the ``config`` parameter of ``tune.Tuner()`` (see example below).

    The content of the ``mlflow`` config entry is used to
    configure MlFlow. Here are the keys you can pass in to this config entry:

    Args:
        tracking_uri: The tracking URI for MLflow tracking. If using
            Tune in a multi-node setting, make sure to use a remote server for
            tracking.
        experiment_id: The id of an already created MLflow experiment.
            All logs from all trials in ``tune.Tuner()`` will be reported to this
            experiment. If this is not provided or the experiment with this
            id does not exist, you must provide an``experiment_name``. This
            parameter takes precedence over ``experiment_name``.
        experiment_name: The name of an already existing MLflow
            experiment. All logs from all trials in ``tune.Tuner()`` will be
            reported to this experiment. If this is not provided, you must
            provide a valid ``experiment_id``.
        token: A token to use for HTTP authentication when
            logging to a remote tracking server. This is useful when you
            want to log to a Databricks server, for example. This value will
            be used to set the MLFLOW_TRACKING_TOKEN environment variable on
            all the remote training processes.

    Example:

    .. code-block:: python

        from ray import tune
        from ray.tune.integration.mlflow import mlflow_mixin

        import mlflow

        # Create the MlFlow expriment.
        mlflow.create_experiment("my_experiment")

        @mlflow_mixin
        def train_fn(config):
            for i in range(10):
                loss = config["a"] + config["b"]
                mlflow.log_metric(key="loss", value=loss)
            tune.report(loss=loss, done=True)

        tuner = tune.Tuner(
            train_fn,
            param_space={
                # define search space here
                "a": tune.choice([1, 2, 3]),
                "b": tune.choice([4, 5, 6]),
                # mlflow configuration
                "mlflow": {
                    "experiment_name": "my_experiment",
                    "tracking_uri": mlflow.get_tracking_uri()
                }
            })

        tuner.fit()

    """
    if ray.util.client.ray.is_connected():
        logger.warning(
            "When using mlflow_mixin with Ray Client, "
            "it is recommended to use a remote tracking "
            "server. If you are using a MLflow tracking server "
            "backed by the local filesystem, then it must be "
            "setup on the server side and not on the client "
            "side."
        )
    if hasattr(func, "__mixins__"):
        func.__mixins__ = func.__mixins__ + (AzureMLflowTrainableMixin,)
    else:
        func.__mixins__ = (AzureMLflowTrainableMixin,)
    return func


class AzureMLflowTrainableMixin:
    "This mixin always grabs default mlflow tracking URI from AzureML workspace."

    def __init__(self, config: Dict, *args, **kwargs):
        if not isinstance(self, Trainable):
            raise ValueError(
                "The `AzureMLflowTrainableMixin` can only be used as a mixin "
                "for `tune.Trainable` classes. Please make sure your "
                "class inherits from both. For example: "
                "`class YourTrainable(AzureMLflowTrainableMixin)`."
            )

        super().__init__(config, *args, **kwargs)
        _config = config.copy()
        try:
            mlflow_config = _config.pop("mlflow").copy()
        except KeyError as e:
            raise ValueError(
                "MLflow mixin specified but no configuration has been passed. "
                "Make sure to include a `mlflow` key in your `config` dict "
                "containing either "
                "`experiment_name` or `experiment_id` specification."
            ) from e

        # Get MLflow Config Info
        ws = Workspace.from_config()
        tracking_uri = ws.get_mlflow_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)

        # Reference to MLFlow Client
        registry_uri = mlflow.get_registry_uri()
        self.mlflow_client = MlflowClient(
            tracking_uri=tracking_uri, registry_uri=registry_uri
        )

        # MLflow Experiment Info
        experiment_id = mlflow_config.pop("experiment_id", None)
        experiment_name = mlflow_config.pop("experiment_name", None)
        if experiment_id is None and experiment_name:
            experiment_id = mlflow.get_experiment_by_name(experiment_name)

        # MLflow Run Info
        nested = mlflow_config.pop("nested", False)
        tags = mlflow_config.pop("tags", None)
        description = mlflow_config.pop("description", None)
        # Use Run Name from Ray Tune's Autogenerated trial_name and trial_id
        run_name = mlflow_config.pop("run_name", None)
        if run_name is None:
            run_name = self.trial_name + "_" + self.trial_id
            run_name = run_name.replace("/", "_")
        tags = {
            **tags,
            "RayTuneRunName": run_name,
            "RayTuneTrialName": self.trial_name,
            "RayTuneTrialID": self.trial_id,
        }

        # Start Run
        run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested,
            tags=tags,
            description=description,
        )
        self.run_id = run.info.run_id

    def stop(self):
        # Specifically end the run created during mixin initialization
        self.mlflow_client.set_terminated(run_id=self.run_id)
