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

The thesis Tables can be re-generated via the sweep configs in `configs/`.

## Citation

If you use this code, please cite the thesis (and the eventual Bioinformatics
paper, once available).

## License

MIT — see [LICENSE](LICENSE).
