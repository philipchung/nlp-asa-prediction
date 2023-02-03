from __future__ import annotations

import logging
from pathlib import Path

import hydra
import mlflow
import numpy as np
from azureml.core import Workspace
from flaml import BlendSearch
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from models.rf.train import train_rf_model_with_mlflow
from models.utils import prepare_datamodule, resolve_paths_and_save_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from ray import air, tune
from ray.tune import CLIReporter, PlacementGroupFactory
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import ExperimentPlateauStopper
from ray.tune.search import ConcurrencyLimiter
from src.metrics import itemize_tensors
from src.modeling.mlflow import set_default_mlflow_tags
from src.modeling.ray import azure_mlflow_tune_mixin
from src.utils import notify, num_cpu

az_http_logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
az_http_logger.setLevel(logging.WARNING)

log = logging.getLogger(__name__)


@azure_mlflow_tune_mixin
def trial(config: dict) -> None:
    "Single hyperparameter tuning trial."
    # Remove extra config elements
    cfg_path = config.pop("cfg_path")
    mlflow_config = config.pop("mlflow")
    # OmegaConf does not support float64 generated from Ray Tune's search algorithms
    config = {
        k: (float(v) if isinstance(v, np.float64) else v) for k, v in config.items()
    }
    # Construct new cfg using selected config for this trial
    with open(cfg_path, "r") as f:
        cfg = OmegaConf.load(f)
    cfg.model.params = config

    # Train Model, Evaluate on Validation Set, Log Metrics
    results = train_rf_model_with_mlflow(cfg)
    val_metrics = results["val_metrics"]
    tags = results["tags"]
    # Report Metrics & MLFlow Info to Tune
    objective_metric = itemize_tensors(
        {k: v for k, v in val_metrics.items() if k in [cfg.tune.objective.metric]}
    )
    report_dict = {
        **objective_metric,
        "mlflow_run_id": tags["RunID"],
        "mlflow_run_name": tags["RunName"],
    }
    tune.report(**report_dict)


@hydra.main(config_path="../conf", config_name="rf_tune_config", version_base=None)
def run_tune(cfg):
    "Tune model on command line using config in `config.yaml`, then overriding tune search spaces."
    # Reproducibility
    seed_everything(seed=cfg.general.seed, workers=True)

    # Configure MLflow
    ws = Workspace.from_config()
    mlflow_uri = ws.get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(mlflow_uri)
    tags = {"ModelType": "RandomForest", "ExperimentType": "Tune"}
    experiment_id = mlflow.create_experiment(
        cfg.mlflow.experiment_name,
        tags=tags,
    )
    with mlflow.start_run(
        experiment_id=experiment_id,
        description="Parent run for many hyperparameter tuning trials.",
    ) as parent_run:
        # Log Tags, Config, Code; Resolve Paths
        set_default_mlflow_tags({**tags, "RunType": "HyperparameterTune"})
        cfg = resolve_paths_and_save_config(cfg)
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True), artifact_file="config.yaml"
        )
        mlflow.log_artifact(Path(__file__).parent.as_posix(), artifact_path="code")
        # Save cfg to disk for child runs to access
        cfg_path = Path(cfg.general.output_dir) / "cfg.yaml"
        with open(cfg_path, "x") as f:
            OmegaConf.save(config=cfg, f=f)

        # Preprocess Data & Generate Caches that Each Tune Trial Can Use
        datamodule = prepare_datamodule(cfg=cfg)

        # Specify Tunable Parameters
        params = {
            "n_estimators": tune.qrandint(**cfg.model.tune.n_estimators),
            "max_samples": tune.uniform(**cfg.model.tune.max_samples),
        }

        # Build Ray-Tune Trial Configuration
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        config = {
            **cfg_dict["model"]["params"],
            **params,
            "cfg_path": cfg_path.as_posix(),
            "mlflow": {
                **cfg.mlflow,
                "experiment_id": experiment_id,
                "run_name": None,  # Ensure we use ray-tune autogenerated name
                "nested": True,
                "tags": {
                    **tags,  # Add experiment tags
                    MLFLOW_PARENT_RUN_ID: parent_run.info.run_id,  # required to nest run
                },
                "description": "Single hyperparameter tuning trial",
            },
        }

        # Configure Tuner
        @notify(webhook_url=cfg.general.notify_webhook_url)
        def hyperparameter_tune_rf():
            scheduler = ASHAScheduler(**cfg.tune.scheduler)
            search_algorithm = ConcurrencyLimiter(
                BlendSearch(),
                max_concurrent=cfg.tune.search.max_concurrent,
            )
            stopper = ExperimentPlateauStopper(**cfg.tune.stopper)
            resources = {
                "CPU": num_cpu(cfg.tune.resource.cpu),
                "GPU": cfg.tune.resource.gpu,
            }
            progress_reporter = CLIReporter(
                metric_columns=[
                    cfg.tune.objective.metric,
                    "mlflow_run_id",
                    "mlflow_run_name",
                ],
                parameter_columns=params.keys(),
                max_report_frequency=cfg.tune.reporter.max_report_frequency,
                sort_by_metric=cfg.tune.reporter.sort_by_metric,
            )

            tuner = tune.Tuner(
                tune.with_resources(
                    trainable=trial, resources=PlacementGroupFactory([resources])
                ),
                param_space=config,
                run_config=air.RunConfig(
                    name=None,
                    local_dir=cfg.general.output_dir,
                    stop=stopper,
                    failure_config=None,
                    sync_config=None,
                    checkpoint_config=None,
                    progress_reporter=progress_reporter,
                    verbose=cfg.tune.params.verbose,
                    log_to_file=True,
                ),
                tune_config=tune.TuneConfig(
                    metric=cfg.tune.objective.metric,
                    mode=cfg.tune.objective.mode,
                    search_alg=search_algorithm,
                    scheduler=scheduler,
                    num_samples=cfg.tune.params.num_trials,
                    max_concurrent_trials=cfg.tune.search.max_concurrent,
                    time_budget_s=None,
                    reuse_actors=cfg.tune.params.reuse_actors,
                ),
            )
            results = tuner.fit()
            # Format Results
            analysis = results._experiment_analysis
            best_dict = analysis.best_result_df.squeeze().to_dict()
            return {
                "ModelType": cfg.model.model_type,
                "Task": cfg.task.task_name,
                f"{cfg.tune.objective.metric}": best_dict[cfg.tune.objective.metric],
                "BestChildMLFlowRunName": best_dict["mlflow_run_name"],
                "BestChildMLFlowRunID": best_dict["mlflow_run_id"],
                "BestRayTrialName": analysis.best_trial,
                "BestChildRayTrialID": analysis.best_result_df.index.item(),
            }

        # Run Tune
        best_trial_info = hyperparameter_tune_rf()
        mlflow.set_tags(best_trial_info)


if __name__ == "__main__":
    run_tune()
