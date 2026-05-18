import copy
import json
from pathlib import Path

import optuna
import torch

from src.multiomics_gnn.config.loader import load_config
from src.multiomics_gnn.pancancer_prediction.experiments.experiment_runner import ExperimentRunner, ExperimentDataLoader


def objective_factory(
    base_cfg: dict,
    fixed_seed: int,
    fixed_split_seed: int,
    fixed_num_gene: int,
):
    # load data
    print("Initializing data loader...")
    exp_data_loader = ExperimentDataLoader(config=base_cfg)
    print("Data loader initialized. Loading data...")
    expression_data, cnv_data, mirna_data = exp_data_loader.load_raw_data()
    print("Data loaded successfully.")
    print(f"Expression data shape: {expression_data.shape}, CNV data shape: {cnv_data.shape}, miRNA data shape: {mirna_data.shape if mirna_data is not None else 'N/A'}")
    runner = ExperimentRunner(expression_data, cnv_data, mirna_data)

    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)

        # ---- fissi per confrontabilità ----
        cfg["project"]["seed"] = fixed_seed
        cfg["project"]["split_seed"] = fixed_split_seed
        cfg["data"]["num_gene"] = fixed_num_gene

        # ---- tuning su 10% del train ----
        cfg.setdefault("tuning", {})
        cfg["tuning"]["train_fraction"] = 0.10

        # disabilita wandb durante tuning
        if "wandb" in cfg:
            cfg["wandb"]["enabled"] = False

        # setup training coerente (tuning baseline)
        cfg["train"]["num_epochs"] = 100
        cfg["early_stopping"]["enabled"] = True
        cfg["early_stopping"]["patience"] = 10
        cfg["early_stopping"]["metric"] = "f1"
        cfg["early_stopping"]["strategy"] = "maximize"

        # ---- search space ----
        cfg["train"]["learning_rate"] = trial.suggest_float("lr", 1e-4, 3e-3, log=True)

        # se vuoi includere anche wd=0.0, devi gestirlo a parte:
        wd_zero = trial.suggest_categorical("wd_zero", [True, False])
        if wd_zero:
            cfg["train"]["weight_decay"] = 0.0
        else:
            cfg["train"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 3e-4, log=True)

        cfg["model"]["dropout"] = trial.suggest_float("dropout", 0.2, 0.8, step=0.2)
        cfg["data"]["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64])
        cfg["model"]["poolsize"] = trial.suggest_categorical("poolsize", [4, 8, 16])

        # feature selection: fissata
        cfg["data"]["feature_selection_method"] = "variance"

        try:
            run_name = f"optuna_trial_{trial.number}"
            summary = runner.run_experiment(cfg, experiment_name=run_name, trial=trial)

            score = summary.get("best_val_metric")

            trial.set_user_attr("run_name", run_name)
            trial.set_user_attr("best_val_metric", score)
            trial.set_user_attr("best_epoch", summary.get("best_epoch"))
            trial.set_user_attr("test_f1", summary.get("test_f1"))
            trial.set_user_attr("test_acc", summary.get("test_acc"))
            trial.set_user_attr("num_gene", fixed_num_gene)
            trial.set_user_attr("batch_size", cfg["data"]["batch_size"])  # <-- CORRETTO

            if score is None or not (score == score):  # None o NaN
                return 0.0
            return float(score)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                return 0.0
            raise

    return objective


def run_study(
    config_path: str,
    n_trials: int,
    timeout_hours: float,
    fixed_seed: int,
    fixed_split_seed: int,
    study_name: str,
    out_dir: str,
    fixed_num_gene: int,
):
    base_cfg = load_config(config_path)

    objective = objective_factory(
        base_cfg=base_cfg,
        fixed_seed=fixed_seed,
        fixed_split_seed=fixed_split_seed,
        fixed_num_gene=fixed_num_gene,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10, n_warmup_steps=20, interval_steps=2
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{(out_dir / (study_name + '.db')).as_posix()}"
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials, timeout=int(timeout_hours * 3600))

    import pandas as pd

    df = study.trials_dataframe(attrs=("number", "state", "value", "params", "user_attrs", "duration"))

    core_cols = [
        "number",
        "user_attrs_run_name",
        "user_attrs_num_gene",
        "user_attrs_batch_size",
        "value",
        "user_attrs_best_val_metric",
        "user_attrs_best_epoch",
        "user_attrs_test_f1",
        "user_attrs_test_acc",
        "state",
        "duration",
    ]
    core_cols = [c for c in core_cols if c in df.columns]
    param_cols = [c for c in df.columns if c.startswith("params_")]

    df = df[core_cols + param_cols].sort_values("value", ascending=False)

    print(f"\n=== OPTUNA REPORT ({study_name}) ===")
    print(df.head(30).to_string(index=False))

    study_dir = out_dir / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(study_dir / "optuna_trials_report.csv", index=False)
    df.to_excel(study_dir / "optuna_trials_report.xlsx", index=False)

    payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "fixed_num_gene": fixed_num_gene,
        "storage": storage,
    }
    (study_dir / "best.json").write_text(json.dumps(payload, indent=2))

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)
    print("Saved best params to:", study_dir / "best.json")
    print("SQLite storage:", storage)

    return study


if __name__ == "__main__":
    num_gene = 700
    run_study(
        config_path="src/multiomics_gnn/config/config.yml",
        n_trials=40,
        timeout_hours=10.0,
        fixed_seed=2025,
        fixed_split_seed=42,
        study_name=f"pancan_ng{num_gene}",
        out_dir="./results/pancan/optuna/" + f"ng{num_gene}",
        fixed_num_gene=num_gene,
    )
