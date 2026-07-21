# MOGNN-TF

**Multi-Omics Graph Neural Network with Transcription Factors for pan-cancer molecular subtype prediction.**

MOGNN-TF integrates gene expression, copy number variation and miRNA expression
from the TCGA Pan-Cancer Atlas into a heterogeneous multi-relational graph,
extending the multimodal GNN framework of Li & Nabavi (2024) with an explicit
transcription factor (TF) layer derived from TFLink. The model is evaluated on
the 27-class iCluster molecular subtype taxonomy of Hoadley et al. (2018).

## Repository layout

```
.
├── configs/                       # YAML configuration files
│   ├── config.yml                 # starting baseline (pre-Optuna)
│   ├── config_final.yml           # paper final config (Optuna trial 18 + TF)
│   └── sweep_*.yml                # 5 ablations × (full / smoke) = 10 files
├── src/multiomics_gnn/            # importable Python package
│   ├── base_ml/                   # Trainer, EarlyStopper, Scheduler, Sampler
│   ├── config/                    # YAML loader
│   ├── models/                    # GAT, GCN, baseline (with TF support)
│   ├── pancancer_prediction/
│   │   ├── datasets/              # OmicsGraphDataset (PyG)
│   │   ├── experiments/           # ExperimentRunner, Optuna objective
│   │   ├── preprocessing/         # feature selection, graph builder
│   │   ├── training/              # build_all factory
│   │   └── utils/                 # stratified splits, metrics
│   └── utils/                     # logger, paths, seeding
├── scripts/                       # CLI entrypoints (see scripts/README.md)
│   ├── train.py
│   ├── run_sweep.py
│   ├── run_optuna.py
│   ├── collect_results.py
│   ├── download_pancan.py
│   ├── analysis/                  # statistical tests
│   └── preprocessing/
│       ├── omics/                 # TCGA omics processing
│       └── priors/                # BioGRID / miRDB / TFLink
├── benchmark/                     # vendored baselines (Li24, MoGCN)
├── supplementary/                 # supplementary material (S1-S6, Figure S1)
│   ├── supplementary.pdf          # as referenced by the manuscript
│   ├── supplementary.tex          # LaTeX source of the PDF
│   └── tables/                    # machine-readable version of every table
├── Makefile                       # pipeline orchestration
├── pyproject.toml                 # package metadata + dependencies
├── environment.yml                # conda environment
└── requirements.txt
```

Generated data live in `data/`, experiment outputs in `results/`,
both gitignored.

## Installation

The reference environment is **Python 3.10 + PyTorch 2.7.0 (CUDA 12.8) +
PyG 2.7.0**. Because PyTorch and the PyG C++ extensions (`torch_scatter`,
`torch_sparse`, `pyg_lib`, …) are CUDA-specific wheels that are *not* on
PyPI, they must be installed from the PyG wheel index **before**
`pip install -e .`.

```bash
# 1. create the env (conda)
conda create -n mognn-tf python=3.10 -y
conda activate mognn-tf

# 2. install the CUDA-12.8 torch / PyG stack from the PyG wheel index
pip install torch==2.7.0 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir \
  pyg_lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# 3. install the package + remaining pure-Python deps
pip install -e .
```

> **Note.** Step 2 installs the CUDA wheels that step 3 (`pip install -e .`)
> would otherwise try to pull from PyPI as CPU-only / wrong-version builds.
> Always run step 2 first. For a CPU-only install, replace `cu128` with
> `cpu` in both URLs. See the
> [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html)
> for other CUDA versions.

## Reproducing the paper

The paper protocol is **Optuna → ablations → analysis**. Two equivalent
workflows are exposed: a *smoke* path (~30 min, sanity-check that the
whole pipeline runs), and a *full* path (matches the paper, hundreds of
GPU-hours).

```bash
make install                 # editable install
make data                    # one-off: TCGA + BioGRID + miRDB + TFLink → data/

# Smoke (quick verification)
make train                   # 1 run with pre-Optuna baseline (configs/config.yml)
make train-final             # 1 run with paper final config (configs/config_final.yml)
make optuna-smoke            # 3 Optuna trials, ~30 min
make ablation-smoke          # ~15 short runs across all 5 ablations
make analysis                # Friedman + Wilcoxon-Holm on whatever is in results/

# Full reproduction (paper)
make optuna                  # 35 trials, ~10h — update config_final.yml with the best
make ablation                # ~450 runs across the 5 thesis tables
make analysis
```

Per-ablation targets are also available (one per paper table):
`make ablation-baseline`, `ablation-genes`, `ablation-sampler`,
`ablation-tf`, `ablation-split-ratio` (each with a matching `-smoke`
variant). Run `make help` for the full list.

> **Note.** `config.yml` is Laterza's starting setup before tuning;
> `config_final.yml` is the result of the Optuna study (trial 18 + the
> full TF layer) and is the configuration reported in the paper tables.
> All ablations are run on top of `config_final.yml`.

The individual preprocessing/training steps are detailed in
[scripts/README.md](scripts/README.md).

### Installing `make` on Windows

`make` is not available out-of-the-box on Windows. Pick one of:

- **Chocolatey** (recommended): `choco install make` (run PowerShell as
  administrator). Requires [Chocolatey](https://chocolatey.org/install)
  installed first.
- **Scoop**: `scoop install make`.
<!-- - **Winget**: `winget install GnuWin32.Make` (then add
  `C:\Program Files (x86)\GnuWin32\bin` to `PATH`).
- **MSYS2 / Git Bash**: `pacman -S make` from an MSYS2 shell. Git for
  Windows ships a `bash` that can run the Makefile if `make` is on `PATH`.
- **WSL**: `wsl --install`, then `sudo apt install make` inside the Ubuntu
  shell and run the commands from there. -->

If you prefer not to install `make`, every target maps 1:1 to a Python
command — see [scripts/README.md](scripts/README.md) and run them
directly, e.g. `python scripts/train.py --config configs/config.yml`
instead of `make train`.

## Supplementary Material

The manuscript refers to Supplementary Tables **S1–S6** and Supplementary
**Figure S1**. They live in [supplementary/](supplementary/), together with a
machine-readable CSV for every table. Labels match the manuscript exactly.

| ID | Content | Machine-readable source |
| --- | --- | --- |
| **Figure S1** | Row-normalised confusion matrices, MOGNN-TF (GCN, 700 genes) vs Li24 (GCN) | figure only — see PDF |
| **S1** | Per-class precision / recall / F1 for the five compared models, mean ± s.d. over eleven seeds | [`S1_per_class_metrics.csv`](supplementary/tables/S1_per_class_metrics.csv) |
| **S2** | Per-class effect of the weighted sampler (γ = 0.98) vs random sampling | [`S2_sampler_per_class.csv`](supplementary/tables/S2_sampler_per_class.csv) |
| **S3** | Effect of the number of selected input genes (*n* ∈ {300, 500, 700, 1000}) | [`S3_gene_ablation.csv`](supplementary/tables/S3_gene_ablation.csv) |
| **S4** | Top transcription factors per iCluster subtype (one-vs-rest ANOVA F-score) | [`S4_tf_top5_per_subtype.csv`](supplementary/tables/S4_tf_top5_per_subtype.csv) · full 27-subtype ranking: [`S4_tf_ranking_full_27subtypes.csv`](supplementary/tables/S4_tf_ranking_full_27subtypes.csv) |
| **S5** | Final MOGNN-TF configuration (Optuna trial 18) | [`S5_hyperparameters.csv`](supplementary/tables/S5_hyperparameters.csv) · [`S5_config_final.yml`](supplementary/tables/S5_config_final.yml) |
| **S6** | Effect of the training fraction (80 / 70 / 60 / 50 %) | [`S6_train_fraction.csv`](supplementary/tables/S6_train_fraction.csv) |

See [supplementary/README.md](supplementary/README.md) for per-table notes and
the pipeline stage that produces each table.

## Data sources

| Modality | Source | Reference |
|---|---|---|
| Gene expression (RNA-seq, EB++ batch-normalized) | UCSC Xena PanCanAtlas | [link](https://xenabrowser.net/datapages/?dataset=EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena&host=https%3A%2F%2Fpancanatlas.xenahubs.net) |
| Copy number (gene-level, GISTIC2) | UCSC Xena PanCanAtlas | [link](https://xenabrowser.net/datapages/?dataset=TCGA.PANCAN.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes&host=https%3A%2F%2Ftcga.xenahubs.net) |
| miRNA expression (EB++) | UCSC Xena PanCanAtlas | [link](https://xenabrowser.net/datapages/?dataset=pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena&host=https%3A%2F%2Fpancanatlas.xenahubs.net) |
| iCluster molecular subtype (27 classes) | UCSC Xena PanCanAtlas, derived from Hoadley *et al.* (2018) | [link](https://xenabrowser.net/datapages/?dataset=TCGA_PanCan33_iCluster_k28_tumor&host=https%3A%2F%2Fpancanatlas.xenahubs.net) |
| Gene–gene interactions | BioGRID 5.0.250 (organism: H. sapiens) | https://thebiogrid.org |
| miRNA–target predictions | miRDB | https://mirdb.org |
| Transcription factor → target | TFLink (H. sapiens, simple format) | https://tflink.net |

After `make data` the directory `data/training/` contains:

```
expression_data_pancan.tsv
cnv_data_pancan.tsv
mirna_data_top_100.tsv
molecular_subtype.csv
adj_matrix_biogrid.npz
standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz
tf_gene_adj_global.npz
tf_nodes_all_in_vocab.csv
```

## Configuration

| File | Purpose |
|---|---|
| [configs/config.yml](configs/config.yml) | Starting baseline used by Laterza before hyperparameter search. `gat_tf`, no TF nodes, `num_epochs=50`, `lr=0.01`, random sampler. |
| [configs/config_final.yml](configs/config_final.yml) | **Paper final configuration** (Optuna trial 18 + TF layer). `gcn_tf`, 700 genes + 100 miRNA + 200 TF, full prior knowledge (`gene_gene` / `mirna_*` / `tf_*` all True), `lr=0.0008`, `dropout=0.2`, `weighted` sampler with `γ=0.98`, `num_epochs=100`, `batch_size=64`. |
| [configs/sweep_*.yml](configs/) | One sweep file per paper table — `baseline`, `genes`, `sampler`, `tf`, `split_ratio` — plus a `*_smoke.yml` variant of each for quick checks. |

All ablations apply their overrides on top of `config_final.yml`.

<!-- ## Citation

If you use this code, please cite the ... -->

## License

MIT — see [LICENSE](LICENSE).
