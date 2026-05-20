# MOGNN-TF — pipeline orchestration
# Each target corresponds to a step in the pipeline described in scripts/README.md.
# Usage:
#   make install          # install dependencies (pip)
#   make data             # full preprocessing: download + omics + priors + training data
#   make train            # single run with default config (MOGNN-TF GCN, 700 genes, 200 TF)
#   make ablation         # sweep: gene count, sampler, TF, split ratio
#   make optuna           # hyperparameter search
#   make analysis         # aggregate results + statistical tests
#   make all              # data -> train -> analysis
#   make clean            # remove caches and intermediate outputs

PY ?= python

.PHONY: help install data data-download data-omics data-priors data-wrap \
        train ablation optuna analysis all clean

help:
	@echo "Targets:"
	@echo "  install   - pip install -e ."
	@echo "  data      - run full preprocessing pipeline"
	@echo "  train     - single MOGNN-TF run with default config"
	@echo "  ablation  - sweep experiments (genes, sampler, TF, split ratio)"
	@echo "  optuna    - hyperparameter search"
	@echo "  analysis  - aggregate metrics + Friedman/Wilcoxon-Holm tests"
	@echo "  all       - data + train + analysis"
	@echo "  clean     - remove __pycache__ and stale logs"

install:
	$(PY) -m pip install -e .

# --------------------------------------------------------------------
# Data preprocessing pipeline
# --------------------------------------------------------------------
data: data-download data-omics data-priors data-wrap

data-download:
	$(PY) scripts/download_pancan.py --out data/raw/tcga_pancan
	$(PY) scripts/preprocessing/omics/prepare_pancan.py \
	    --src data/raw/tcga_pancan \
	    --out data/raw/tcga_pancan/processed

# Note: no --normalization here. The wrapper / training pipeline performs
# min-max scaling on the training fold only (to avoid data leakage), so
# applying it pan-cohort upstream is redundant and produces files with a
# "_normalized" suffix that data-wrap does not expect.
data-omics:
	$(PY) scripts/preprocessing/omics/process_data_pancan.py

data-priors:
	$(PY) scripts/preprocessing/priors/get_raw_data.py
	$(PY) scripts/preprocessing/priors/refseq2gene.py
	$(PY) scripts/preprocessing/priors/load_interaction.py

data-wrap:
	$(PY) scripts/preprocessing/omics/data_wrapper.py

# --------------------------------------------------------------------
# Training & evaluation
# --------------------------------------------------------------------
train:
	$(PY) scripts/train.py --config configs/config.yml

ablation:
	$(PY) scripts/run_sweep.py --sweep configs/sweep_config.yml

optuna:
	$(PY) scripts/run_optuna.py

analysis:
	$(PY) scripts/collect_results.py --results-root results/pancan
	$(PY) scripts/analysis/analize_metrics.py

all: data train analysis

# --------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------
clean:
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
	@rm -f pipeline.log
