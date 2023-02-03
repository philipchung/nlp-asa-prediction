import hydra
from models.bert.tune import run_tune


@hydra.main(
    config_path="../conf",
    config_name="bioclinicalbert_tune_config",
    version_base=None,
)
def run_tune_bioclinicalbert(cfg):
    run_tune(cfg)


if __name__ == "__main__":
    run_tune_bioclinicalbert()
