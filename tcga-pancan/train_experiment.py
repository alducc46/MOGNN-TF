import sys
import os
from src.multiomics_gnn.config.loader import load_config
# Make imports work regardless of current working directory
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = HERE
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.multiomics_gnn.pancancer_prediction.experiments.experiment_runner import ExperimentRunner


if __name__ == "__main__":
    config_path = os.path.join(REPO_ROOT, "src", "multiomics_gnn", "config", "config.yml")

    runner = ExperimentRunner()
    cfg = load_config(config_path)
    runner.run_experiment(cfg)
