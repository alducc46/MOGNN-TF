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
│   ├── config.yml                 # default MOGNN-TF setup (GCN, 700 genes, 200 TF, γ=0.98)
│   ├── sweep_config.yml           # ablation sweeps
│   ├── sweep_config_grid.yml
│   └── sweep_decoder.yml
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
├── benchmark/                     # baseline implementations (Li24, MoGCN, ...)
├── Makefile                       # pipeline orchestration
├── pyproject.toml                 # package metadata + dependencies
├── environment.yml                # conda environment
└── requirements.txt
```

Generated data live in `data/`, experiment outputs in `results/`,
both gitignored.

## Installation

Option A — conda:

```bash
conda env create -f environment.yml
conda activate mognn-tf
pip install -e .
```

Option B — venv + pip:

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
```

> **Note on PyTorch / PyG.** `pyproject.toml` pins `torch==2.1.1` and
> `torch_geometric==2.4.0`. If you need CUDA wheels for `torch_scatter`,
> `torch_sparse`, etc., install them separately from the PyG wheel index
> after installing torch (see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html)).


If `torch-sparse` installation fails (common on some platforms), install PyTorch and PyG extensions separately:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install --no-cache-dir --only-binary=:all: \
  pyg_lib torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv \
  termcolor torcheval \
  -f https://data.pyg.org/whl/torch-2.7.1+cu128.html


## Reproducing the thesis results

End-to-end reproduction in five commands:

```bash
make install     # editable install
make data        # download TCGA + BioGRID + miRDB + TFLink, build all matrices
make train       # single run with the final MOGNN-TF (GCN) configuration
make ablation    # gene count / sampler / TF / split-ratio ablations
make analysis    # aggregate metrics, Friedman + Wilcoxon-Holm tests
```

`make all` runs `data → train → analysis` sequentially.

The individual steps are detailed in [scripts/README.md](scripts/README.md).

### Installing `make` on Windows

`make` is not available out-of-the-box on Windows. Pick one of:

- **Chocolatey** (recommended): `choco install make` (run PowerShell as
  administrator). Requires [Chocolatey](https://chocolatey.org/install)
  installed first.
- **Scoop**: `scoop install make`.
- **Winget**: `winget install GnuWin32.Make` (then add
  `C:\Program Files (x86)\GnuWin32\bin` to `PATH`).
- **MSYS2 / Git Bash**: `pacman -S make` from an MSYS2 shell. Git for
  Windows ships a `bash` that can run the Makefile if `make` is on `PATH`.
- **WSL**: `wsl --install`, then `sudo apt install make` inside the Ubuntu
  shell and run the commands from there.

If you prefer not to install `make`, every target maps 1:1 to a Python
command — see [scripts/README.md](scripts/README.md) and run them
directly, e.g. `python scripts/train.py --config configs/config.yml`
instead of `make train`.

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

Default configuration in [configs/config.yml](configs/config.yml). Key fields:

- `data.num_gene = 700`, `data.num_mirna = 100`, `data.num_tf = 200`
- `data.omic_mode = 4` (expression + CNV + miRNA)
- `data.gene_gene = True`, `data.mirna_gene = True`, `data.mirna_mirna = True`,
  `data.tf_gene = True`, `data.tf_tf = True`, `data.tf_mirna = True`
- `model.name = gcn_tf` (or `gat_tf`)
- `sampler_strategy = weighted`, `sampler_gamma = 0.98`
- `train.num_epochs = 100`, early stopping on val macro-F1

The ablation results can be re-generated via the sweep configs in `configs/`.

<!-- ## Citation

If you use this code, please cite the ... -->

## License

MIT — see [LICENSE](LICENSE).
