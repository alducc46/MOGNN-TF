# Supplementary Material

Supplementary material for:

> **MOGNN-TF: a heterogeneous graph neural network integrating transcription-factor
> regulation for pan-cancer molecular subtype classification**

The manuscript refers to Supplementary Tables **S1–S6** and Supplementary
**Figure S1**. This directory is the authoritative home of that material: the
compiled PDF plus a machine-readable CSV for every table.

Labels here match the manuscript exactly — `S1` in the paper is `S1` here.

All tables report results produced by the pipeline in this repository, run on the
TCGA Pan-Cancer Atlas cohort under `configs/config_final.yml`. Per-class and
ablation figures are means ± s.d. over eleven seeds on the test set.

## Contents

| File | Description |
| --- | --- |
| [`supplementary.pdf`](supplementary.pdf) | The supplementary document as referenced by the manuscript (S1–S6, Figure S1) |
| [`supplementary.tex`](supplementary.tex) | LaTeX source of the document |
| [`tables/`](tables/) | Machine-readable version of every table (see below) |

## Tables

| ID | Content | CSV | Produced by |
| --- | --- | --- | --- |
| — | iCluster subtype legend (C1–C28) | [`subtype_legend.csv`](tables/subtype_legend.csv) | taxonomy of Hoadley *et al.* (2018) |
| **Figure S1** | Row-normalised confusion matrices, MOGNN-TF (GCN, 700 genes) vs Li24 (GCN) | figure only — see PDF | per-run `confusion_matrix_cm.csv`, averaged over seeds |
| **S1** | Per-class precision / recall / F1, five compared models | [`S1_per_class_metrics.csv`](tables/S1_per_class_metrics.csv) | `make ablation-baseline` → `make analysis` |
| **S2** | Per-class effect of the weighted sampler (γ = 0.98) vs random sampling | [`S2_sampler_per_class.csv`](tables/S2_sampler_per_class.csv) | `make ablation-sampler` → `make analysis` |
| **S3** | Effect of the number of selected input genes (*n* ∈ {300, 500, 700, 1000}) | [`S3_gene_ablation.csv`](tables/S3_gene_ablation.csv) | `make ablation-genes` → `make analysis` |
| **S4** | Transcription factors per iCluster subtype (one-vs-rest ANOVA F-score) | [`S4_tf_top5_per_subtype.csv`](tables/S4_tf_top5_per_subtype.csv) · [`S4_tf_ranking_full_27subtypes.csv`](tables/S4_tf_ranking_full_27subtypes.csv) · [`S4_tf_ranking_global.csv`](tables/S4_tf_ranking_global.csv) | `scripts/analysis/tf_fscore_ranking.py` |
| **S5** | Final MOGNN-TF configuration (Optuna trial 18) | [`S5_hyperparameters.csv`](tables/S5_hyperparameters.csv) · [`S5_config_final.yml`](tables/S5_config_final.yml) | `make optuna` |
| **S6** | Effect of the training fraction (80 / 70 / 60 / 50 %) | [`S6_train_fraction.csv`](tables/S6_train_fraction.csv) | `make ablation-split-ratio` → `make analysis` |

Aggregation across runs is done by `scripts/collect_results.py`; the Wilcoxon
signed-rank tests with Holm correction reported in S3 come from
`scripts/analysis/analize_metrics.py`. Both are invoked by `make analysis`.
See the [main README](../README.md#reproducing-the-paper) for the full protocol.

### Notes on individual tables

**S1.** Emitted in long (tidy) format: one row per `class × model × metric`
(27 × 5 × 3 = 405 rows). The `is_best` flag marks the best value for that metric
within that class. Cluster **C24** (LAML) is absent from the per-class
evaluation, so class numbering skips from C23 to C25 — this is expected, not an
omission.

**S3, S6.** `GCN (700)` is the reference configuration against which the Wilcoxon
tests in S3 are computed, so its `p_value` and `p_holm` fields are empty.

**S4.** The PDF prints the top-5 TFs for the ten subtypes with the cleanest
lineage signal, and its caption states that the full ranking is released with the
project code. That full ranking is
[`S4_tf_ranking_full_27subtypes.csv`](tables/S4_tf_ranking_full_27subtypes.csv):
all 27 subtypes × 200 TFs (5,400 rows), with `f_score`, `p_value` and
`rank_in_subtype`. Taking the top 5 per subtype from it reproduces the ten
subtypes printed in the PDF exactly.
[`S4_tf_ranking_global.csv`](tables/S4_tf_ranking_global.csv) additionally gives
the cross-subtype aggregate ranking.

**S5.** [`S5_config_final.yml`](tables/S5_config_final.yml) is the configuration
consumed by the training pipeline (`configs/config_final.yml`); the values in
Table S5 are reproduced verbatim by it.

## Reproducing these results

The repository ships the complete pipeline — preprocessing, model, sweep
configurations and analysis — so the experiments behind these tables can be re-run
end to end. See [Reproducing the paper](../README.md#reproducing-the-paper).

Note that the full protocol is on the order of hundreds of GPU-hours, and that
exact numerical agreement additionally depends on the CUDA/PyTorch build, since
GPU reductions are not bitwise deterministic across environments.
