import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


def get_config_columns(df: pd.DataFrame) -> list[str]:
    exclude_cols = {
        "experiment_name",
        "seed",
        "split_seed",
        "timestamp",
        "file_path",
        "accuracy",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "precision",
        "recall",
        "f1",
        "support",
        "class_idx",
        "class_name",
        "config_id",
    }
    return [c for c in df.columns if c not in exclude_cols]


def make_config_id(df: pd.DataFrame, config_columns: list[str] | None = None) -> pd.Series:
    if config_columns is None:
        config_columns = get_config_columns(df)

    if len(config_columns) == 0:
        return pd.Series(["config=default"] * len(df), index=df.index)

    parts = []
    for col in config_columns:
        parts.append(col + "=" + df[col].fillna("NA").astype(str))
    return pd.Series(parts[0], index=df.index).str.cat(parts[1:], sep="|") if len(parts) > 1 else parts[0]


def cliffs_delta_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    pos = np.sum(diff > 0)
    neg = np.sum(diff < 0)
    n = len(diff)
    return (pos - neg) / n if n > 0 else np.nan


def run_tests(
    csv_path: Path,
    metric: str = "f1_macro",
    alpha: float = 0.05,
    correction: str = "holm",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    required_cols = {"seed", metric}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")

    df = df.copy()
    cfg_cols = get_config_columns(df)
    df["config_id"] = make_config_id(df, cfg_cols)

    pivot = (
        df.pivot_table(index="seed", columns="config_id", values=metric, aggfunc="mean")
        .sort_index(axis=1)
    )
    pivot = pivot.dropna(axis=0, how="any")

    if pivot.shape[0] < 3:
        raise ValueError(
            "Troppi seed mancanti tra configurazioni: servono almeno 3 seed comuni per il confronto."
        )

    # Test omnibus non parametrico per >2 configurazioni (misure ripetute sui seed)
    friedman_stat, friedman_p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])

    pairwise_rows = []
    for cfg_a, cfg_b in itertools.combinations(pivot.columns, 2):
        x = pivot[cfg_a].values
        y = pivot[cfg_b].values
        w_stat, p_val = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        delta = cliffs_delta_paired(x, y)
        pairwise_rows.append(
            {
                "metric": metric,
                "config_a": cfg_a,
                "config_b": cfg_b,
                "n_pairs": len(x),
                "mean_a": float(np.mean(x)),
                "mean_b": float(np.mean(y)),
                "mean_diff_a_minus_b": float(np.mean(x - y)),
                "wilcoxon_W": float(w_stat),
                "p_value": float(p_val),
                "paired_cliffs_delta": float(delta),
            }
        )

    pairwise_df = pd.DataFrame(pairwise_rows)
    reject, p_adj, _, _ = multipletests(pairwise_df["p_value"].values, alpha=alpha, method=correction)
    pairwise_df["p_value_adj"] = p_adj
    pairwise_df["significant_adj"] = reject
    pairwise_df = pairwise_df.sort_values("p_value_adj", ascending=True).reset_index(drop=True)

    ranks = pivot.apply(lambda row: rankdata(-row), axis=1, result_type="expand")
    ranks.columns = pivot.columns
    mean_rank = ranks.mean(axis=0)
    mean_scores = pivot.mean(axis=0)

    summary_df = pd.DataFrame(
        {
            "metric": metric,
            "config_id": mean_scores.index,
            "n_common_seeds": pivot.shape[0],
            "mean_score": mean_scores.values,
            "mean_rank_friedman": mean_rank.values,
            "friedman_stat": friedman_stat,
            "friedman_p_value": friedman_p,
        }
    ).sort_values(["mean_score", "mean_rank_friedman"], ascending=[False, True])

    return summary_df, pairwise_df


def choose_baseline_config(summary_df: pd.DataFrame) -> str:
    cfgs = summary_df["config_id"].astype(str)

    preferred_gg_mm = cfgs[
        cfgs.str.contains("gene_gene=True", regex=False)
        & cfgs.str.contains("mirna_mirna=True", regex=False)
    ]
    if len(preferred_gg_mm) > 0:
        preferred_rows = summary_df[summary_df["config_id"].isin(preferred_gg_mm)]
        return preferred_rows.sort_values("mean_score", ascending=True).iloc[0]["config_id"]

    preferred = cfgs[
        cfgs.str.contains("mode=0", regex=False)
        & cfgs.str.contains("num_mirna=0", regex=False)
        & cfgs.str.contains("num_tf=0", regex=False)
    ]
    if len(preferred) > 0:
        return preferred.iloc[0]

    return summary_df.sort_values("mean_score", ascending=True).iloc[0]["config_id"]


def run_topk_vs_baseline_tests(
    csv_path: Path,
    metric: str,
    top_k: int = 3,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    required_cols = {"seed", metric}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")

    cfg_cols = get_config_columns(df)
    df["config_id"] = make_config_id(df, cfg_cols)

    pivot = (
        df.pivot_table(index="seed", columns="config_id", values=metric, aggfunc="mean")
        .sort_index(axis=1)
        .dropna(axis=0, how="any")
    )
    if pivot.shape[0] < 3:
        raise ValueError("Servono almeno 3 seed comuni per i test top-vs-baseline.")

    mean_scores = pivot.mean(axis=0).sort_values(ascending=False)
    summary_df = pd.DataFrame({"config_id": mean_scores.index, "mean_score": mean_scores.values})

    baseline_cfg = choose_baseline_config(summary_df)
    top_cfgs = [cfg for cfg in mean_scores.index.tolist() if cfg != baseline_cfg][:top_k]

    if len(top_cfgs) == 0:
        raise ValueError("Nessuna configurazione top disponibile diversa dal baseline.")

    rows = []
    base_vals = pivot[baseline_cfg].values
    for cfg in top_cfgs:
        x = pivot[cfg].values
        diff = x - base_vals

        if np.allclose(diff, 0.0):
            w_stat, p_value = 0.0, 1.0
        else:
            w_stat, p_value = wilcoxon(x, base_vals, alternative=alternative, zero_method="wilcox")

        rows.append(
            {
                "metric": metric,
                "alternative": alternative,
                "baseline_config": baseline_cfg,
                "candidate_config": cfg,
                "n_pairs": int(len(x)),
                "mean_candidate": float(np.mean(x)),
                "mean_baseline": float(np.mean(base_vals)),
                "mean_diff_candidate_minus_baseline": float(np.mean(diff)),
                "wins_candidate": int(np.sum(diff > 0)),
                "ties": int(np.sum(diff == 0)),
                "losses_candidate": int(np.sum(diff < 0)),
                "paired_cliffs_delta": float(cliffs_delta_paired(x, base_vals)),
                "wilcoxon_W": float(w_stat),
                "p_value": float(p_value),
            }
        )

    out_df = pd.DataFrame(rows)
    holm_rej, holm_p, _, _ = multipletests(out_df["p_value"].values, alpha=alpha, method="holm")
    fdr_rej, fdr_p, _, _ = multipletests(out_df["p_value"].values, alpha=alpha, method="fdr_bh")
    out_df["p_value_holm"] = holm_p
    out_df["significant_holm"] = holm_rej
    out_df["p_value_fdr_bh"] = fdr_p
    out_df["significant_fdr_bh"] = fdr_rej

    out_df = out_df.sort_values(["p_value_fdr_bh", "mean_diff_candidate_minus_baseline"], ascending=[True, False])
    return out_df.reset_index(drop=True)


def summarize_topk_vs_baseline(top_df: pd.DataFrame, metric: str) -> dict:
    n_total = len(top_df)
    n_holm = int(top_df["significant_holm"].sum())
    n_fdr = int(top_df["significant_fdr_bh"].sum())

    best_row = top_df.sort_values(["p_value_fdr_bh", "mean_diff_candidate_minus_baseline"], ascending=[True, False]).iloc[0]
    return {
        "metric": metric,
        "n_candidates_tested": n_total,
        "n_significant_holm": n_holm,
        "n_significant_fdr_bh": n_fdr,
        "overall_holm": "significant" if n_holm > 0 else "not_significant",
        "overall_fdr_bh": "significant" if n_fdr > 0 else "not_significant",
        "best_candidate_config": best_row["candidate_config"],
        "baseline_config": best_row["baseline_config"],
        "best_mean_diff": float(best_row["mean_diff_candidate_minus_baseline"]),
        "best_p_holm": float(best_row["p_value_holm"]),
        "best_p_fdr_bh": float(best_row["p_value_fdr_bh"]),
    }


def _pick_config_by_rules(mean_scores: pd.Series, rules: list[str]) -> str | None:
    cfgs = mean_scores.index.astype(str)
    mask = np.ones(len(cfgs), dtype=bool)
    for rule in rules:
        mask &= pd.Series(cfgs).str.contains(rule, regex=False).values
    selected = mean_scores[pd.Series(mask, index=mean_scores.index)]
    if len(selected) == 0:
        return None
    return selected.sort_values(ascending=False).index[0]


def run_genes_mirna_vs_tf_test(
    csv_path: Path,
    metric: str,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    required_cols = {"seed", metric}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {sorted(missing)}")

    cfg_cols = get_config_columns(df)
    df["config_id"] = make_config_id(df, cfg_cols)

    pivot = (
        df.pivot_table(index="seed", columns="config_id", values=metric, aggfunc="mean")
        .sort_index(axis=1)
        .dropna(axis=0, how="any")
    )
    if pivot.shape[0] < 3:
        raise ValueError("Servono almeno 3 seed comuni per il confronto baseline vs TF.")

    mean_scores = pivot.mean(axis=0).sort_values(ascending=False)

    baseline_rules_strict = [
        "num_mirna=100",
        "num_tf=0",
        "gene_gene=True",
        "mirna_gene=True",
        "mirna_mirna=True",
        "tf_gene=False",
        "tf_mirna=False",
        "tf_tf=False",
        "mode=4",
    ]
    baseline_rules_fallback = baseline_rules_strict[:-1]

    tf_rules_strict = [
        "num_mirna=100",
        "num_tf=200",
        "gene_gene=True",
        "mirna_gene=True",
        "mirna_mirna=True",
        "tf_gene=True",
        "tf_mirna=True",
        "tf_tf=True",
        "mode=4",
    ]
    tf_rules_fallback = tf_rules_strict[:-1]

    baseline_cfg = _pick_config_by_rules(mean_scores, baseline_rules_strict)
    if baseline_cfg is None:
        baseline_cfg = _pick_config_by_rules(mean_scores, baseline_rules_fallback)

    tf_cfg = _pick_config_by_rules(mean_scores, tf_rules_strict)
    if tf_cfg is None:
        tf_cfg = _pick_config_by_rules(mean_scores, tf_rules_fallback)

    if baseline_cfg is None or tf_cfg is None:
        raise ValueError("Impossibile trovare baseline geni+miRNA o configurazione con TF nei risultati.")

    paired = pivot[[tf_cfg, baseline_cfg]].dropna(axis=0, how="any")
    x = paired[tf_cfg].values
    y = paired[baseline_cfg].values
    diff = x - y

    if np.allclose(diff, 0.0):
        w_stat, p_value = 0.0, 1.0
    else:
        w_stat, p_value = wilcoxon(x, y, alternative=alternative, zero_method="wilcox")

    result = {
        "metric": metric,
        "alternative": alternative,
        "baseline_genes_mirna": baseline_cfg,
        "candidate_with_tf": tf_cfg,
        "n_pairs": int(len(diff)),
        "mean_tf": float(np.mean(x)),
        "mean_baseline": float(np.mean(y)),
        "mean_diff_tf_minus_baseline": float(np.mean(diff)),
        "wins_tf": int(np.sum(diff > 0)),
        "ties": int(np.sum(diff == 0)),
        "losses_tf": int(np.sum(diff < 0)),
        "paired_cliffs_delta": float(cliffs_delta_paired(x, y)),
        "wilcoxon_W": float(w_stat),
        "p_value": float(p_value),
        "significant_0_05": bool(p_value < alpha),
    }
    return pd.DataFrame([result])


def run_per_class_tests(
    csv_path: Path,
    alpha: float = 0.05,
    correction: str = "holm",
    min_pairs: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    required_cols = {"class_idx", "class_name", "seed", "f1"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti nel CSV per-classe: {sorted(missing)}")

    df = df.copy()
    cfg_cols = get_config_columns(df)
    df["config_id"] = make_config_id(df, cfg_cols)

    pairwise_rows = []
    class_rows = []

    for (class_idx, class_name), class_df in df.groupby(["class_idx", "class_name"], dropna=False):
        pivot = (
            class_df.pivot_table(index="seed", columns="config_id", values="f1", aggfunc="mean")
            .sort_index(axis=1)
            .dropna(axis=0, how="any")
        )

        n_cfg = pivot.shape[1]
        n_pairs = pivot.shape[0]

        if n_cfg < 2 or n_pairs < min_pairs:
            continue

        if n_cfg >= 3:
            friedman_stat, friedman_p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
        else:
            friedman_stat, friedman_p = np.nan, np.nan

        means = pivot.mean(axis=0).sort_values(ascending=False)
        best_cfg = means.index[0]

        class_rows.append(
            {
                "class_idx": int(class_idx),
                "class_name": class_name,
                "n_common_seeds": int(n_pairs),
                "n_configurations": int(n_cfg),
                "best_config": best_cfg,
                "best_mean_f1": float(means.iloc[0]),
                "second_mean_f1": float(means.iloc[1]) if len(means) > 1 else np.nan,
                "gap_best_second": float(means.iloc[0] - means.iloc[1]) if len(means) > 1 else np.nan,
                "friedman_stat": friedman_stat,
                "friedman_p_value": friedman_p,
            }
        )

        for cfg_a, cfg_b in itertools.combinations(pivot.columns, 2):
            x = pivot[cfg_a].values
            y = pivot[cfg_b].values
            w_stat, p_val = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
            diff = x - y
            pairwise_rows.append(
                {
                    "class_idx": int(class_idx),
                    "class_name": class_name,
                    "config_a": cfg_a,
                    "config_b": cfg_b,
                    "n_pairs": len(x),
                    "mean_a": float(np.mean(x)),
                    "mean_b": float(np.mean(y)),
                    "mean_diff_a_minus_b": float(np.mean(diff)),
                    "wilcoxon_W": float(w_stat),
                    "p_value": float(p_val),
                    "paired_cliffs_delta": float(cliffs_delta_paired(x, y)),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    class_summary_df = pd.DataFrame(class_rows)

    if pairwise_df.empty:
        raise ValueError("Nessun confronto per-classe disponibile: controlla seed comuni e configurazioni.")

    pairwise_df["p_value_adj"] = np.nan
    pairwise_df["significant_adj"] = False

    for cls, idx in pairwise_df.groupby("class_name").groups.items():
        pvals = pairwise_df.loc[idx, "p_value"].values
        reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method=correction)
        pairwise_df.loc[idx, "p_value_adj"] = p_adj
        pairwise_df.loc[idx, "significant_adj"] = reject

    if not class_summary_df.empty:
        class_summary_df["n_significant_pairs"] = 0
        class_summary_df["n_sig_wins_best"] = 0
        class_summary_df["n_sig_losses_best"] = 0

        for i, row in class_summary_df.iterrows():
            cls = row["class_name"]
            best_cfg = row["best_config"]
            class_pairs = pairwise_df[pairwise_df["class_name"] == cls]
            sig = class_pairs[class_pairs["significant_adj"]]

            wins = 0
            losses = 0
            for _, r in sig.iterrows():
                if r["config_a"] == best_cfg:
                    if r["mean_diff_a_minus_b"] > 0:
                        wins += 1
                    elif r["mean_diff_a_minus_b"] < 0:
                        losses += 1
                elif r["config_b"] == best_cfg:
                    if r["mean_diff_a_minus_b"] < 0:
                        wins += 1
                    elif r["mean_diff_a_minus_b"] > 0:
                        losses += 1

            class_summary_df.loc[i, "n_significant_pairs"] = int(len(sig))
            class_summary_df.loc[i, "n_sig_wins_best"] = int(wins)
            class_summary_df.loc[i, "n_sig_losses_best"] = int(losses)

    class_summary_df = class_summary_df.sort_values(
        ["n_sig_wins_best", "gap_best_second"],
        ascending=[False, False],
    ).reset_index(drop=True)

    pairwise_df = pairwise_df.sort_values(
        ["class_name", "p_value_adj", "p_value"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    significant_only_df = pairwise_df[pairwise_df["significant_adj"]].copy()

    return class_summary_df, pairwise_df, significant_only_df


def run_best_vs_baseline_per_class(
    detailed_csv_path: Path,
    class_summary_df: pd.DataFrame,
    alpha: float = 0.05,
    correction: str = "holm",
    min_pairs: int = 3,
) -> pd.DataFrame:
    df = pd.read_csv(detailed_csv_path).copy()
    cfg_cols = get_config_columns(df)
    df["config_id"] = make_config_id(df, cfg_cols)

    if "sampler_strategy" not in df.columns:
        raise ValueError("Colonna sampler_strategy non presente: baseline random non disponibile.")

    random_cfgs = df.loc[df["sampler_strategy"].astype(str) == "random", "config_id"].dropna().unique().tolist()
    if len(random_cfgs) == 0:
        raise ValueError("Configurazione baseline random non trovata nei dati.")
    baseline_config = random_cfgs[0]
    rows = []

    for _, cls_row in class_summary_df.iterrows():
        class_name = cls_row["class_name"]
        class_idx = int(cls_row["class_idx"])
        best_cfg = cls_row["best_config"]

        class_df = df[df["class_name"] == class_name]
        pivot = (
            class_df.pivot_table(index="seed", columns="config_id", values="f1", aggfunc="mean")
            .sort_index(axis=1)
        )

        if baseline_config not in pivot.columns:
            continue

        means = pivot.mean(axis=0).sort_values(ascending=False)
        if best_cfg == baseline_config:
            non_baseline = [cfg for cfg in means.index.tolist() if cfg != baseline_config]
            if not non_baseline:
                continue
            best_cfg = non_baseline[0]

        if best_cfg not in pivot.columns:
            continue

        paired = pivot[[best_cfg, baseline_config]].dropna(axis=0, how="any")
        if paired.shape[0] < min_pairs:
            continue

        best_vals = paired[best_cfg].values
        base_vals = paired[baseline_config].values
        diff = best_vals - base_vals

        wins = int(np.sum(diff > 0))
        losses = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))

        if np.allclose(diff, 0.0):
            w_stat, p_value = 0.0, 1.0
        else:
            w_stat, p_value = wilcoxon(best_vals, base_vals, alternative="two-sided", zero_method="wilcox")

        rows.append(
            {
                "class_idx": class_idx,
                "class_name": class_name,
                "baseline_config": baseline_config,
                "best_config": best_cfg,
                "n_pairs": int(len(diff)),
                "mean_f1_best": float(np.mean(best_vals)),
                "mean_f1_baseline": float(np.mean(base_vals)),
                "mean_diff_best_minus_baseline": float(np.mean(diff)),
                "wins_best": wins,
                "ties": ties,
                "losses_best": losses,
                "win_rate_best": float(wins / len(diff)),
                "paired_cliffs_delta": float(cliffs_delta_paired(best_vals, base_vals)),
                "wilcoxon_W": float(w_stat),
                "p_value": float(p_value),
            }
        )

    report_df = pd.DataFrame(rows)
    if report_df.empty:
        raise ValueError("Nessun confronto best-vs-baseline disponibile.")

    reject, p_adj, _, _ = multipletests(report_df["p_value"].values, alpha=alpha, method=correction)
    report_df["p_value_adj"] = p_adj
    report_df["significant_adj"] = reject

    report_df = report_df.sort_values(
        ["p_value_adj", "mean_diff_best_minus_baseline"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return report_df


def plot_best_vs_baseline_heatmap(report_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = report_df.sort_values("mean_diff_best_minus_baseline", ascending=True).copy()

    values = plot_df["mean_diff_best_minus_baseline"].values.reshape(-1, 1)
    labels = plot_df["class_name"].tolist()

    fig_height = max(6, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(6, fig_height))

    vmax = np.max(np.abs(values)) if np.max(np.abs(values)) > 0 else 1e-6
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0])
    ax.set_xticklabels(["Best - Baseline\n(mean F1)"])
    ax.set_title("Per-class improvement vs random baseline")

    for i, row in plot_df.reset_index(drop=True).iterrows():
        val = row["mean_diff_best_minus_baseline"]
        mark = "*" if row["significant_adj"] else ""
        ax.text(0, i, f"{val:+.4f}{mark}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ΔF1 (best - baseline)")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    csv_file = base_dir / "experiments_summary.csv"
    topk_summary_rows = []

    for metric_name in ["f1_macro", "f1_weighted"]:
        summary, pairwise = run_tests(
            csv_path=csv_file,
            metric=metric_name,
            alpha=0.05,
            correction="holm",
        )

        summary_out = base_dir / f"wilcoxon_{metric_name}_summary.csv"
        pairwise_out = base_dir / f"wilcoxon_{metric_name}_pairwise.csv"
        summary.to_csv(summary_out, index=False)
        pairwise.to_csv(pairwise_out, index=False)

        print(f"\n=== {metric_name} ===")
        print(summary[["config_id", "mean_score", "mean_rank_friedman", "friedman_p_value"]])
        print(f"\nSalvati: {summary_out.name}, {pairwise_out.name}")

        top_vs_base = run_topk_vs_baseline_tests(
            csv_path=csv_file,
            metric=metric_name,
            top_k=3,
            alternative="greater",
            alpha=0.05,
        )
        top_vs_base_out = base_dir / f"wilcoxon_{metric_name}_top3_vs_baseline.csv"
        top_vs_base.to_csv(top_vs_base_out, index=False)
        topk_summary_rows.append(summarize_topk_vs_baseline(top_vs_base, metric_name))
        print(f"Top-3 vs baseline salvato: {top_vs_base_out.name}")

        genes_mirna_vs_tf = run_genes_mirna_vs_tf_test(
            csv_path=csv_file,
            metric=metric_name,
            alternative="greater",
            alpha=0.05,
        )
        genes_mirna_vs_tf_out = base_dir / f"wilcoxon_{metric_name}_genes_mirna_vs_tf.csv"
        genes_mirna_vs_tf.to_csv(genes_mirna_vs_tf_out, index=False)
        print(f"Confronto baseline geni+miRNA vs TF salvato: {genes_mirna_vs_tf_out.name}")

    if len(topk_summary_rows) > 0:
        topk_summary_df = pd.DataFrame(topk_summary_rows)
        topk_summary_out = base_dir / "wilcoxon_top3_vs_baseline_interpretation.csv"
        topk_summary_df.to_csv(topk_summary_out, index=False)
        print(f"Report interpretativo salvato: {topk_summary_out.name}")

    per_class_file = base_dir / "per_class_metrics_detailed.csv"
    if per_class_file.exists():
        class_summary, class_pairwise, class_significant = run_per_class_tests(
            csv_path=per_class_file,
            alpha=0.05,
            correction="holm",
            min_pairs=3,
        )

        class_summary_out = base_dir / "wilcoxon_per_class_summary.csv"
        class_pairwise_out = base_dir / "wilcoxon_per_class_pairwise.csv"
        class_significant_out = base_dir / "wilcoxon_per_class_significant_only.csv"

        class_summary.to_csv(class_summary_out, index=False)
        class_pairwise.to_csv(class_pairwise_out, index=False)
        class_significant.to_csv(class_significant_out, index=False)

        print("\n=== Per-class F1 (Wilcoxon pairwise) ===")
        print(class_summary[["class_name", "best_mean_f1", "gap_best_second", "n_significant_pairs", "n_sig_wins_best"]].head(10))
        print(
            f"\nSalvati: {class_summary_out.name}, {class_pairwise_out.name}, {class_significant_out.name}"
        )

        try:
            best_vs_baseline_df = run_best_vs_baseline_per_class(
                detailed_csv_path=per_class_file,
                class_summary_df=class_summary,
                alpha=0.05,
                correction="holm",
                min_pairs=3,
            )
            best_vs_baseline_out = base_dir / "wilcoxon_best_vs_random_per_class.csv"
            best_vs_baseline_df.to_csv(best_vs_baseline_out, index=False)

            heatmap_out = base_dir / "best_vs_random_per_class_heatmap.png"
            plot_best_vs_baseline_heatmap(best_vs_baseline_df, heatmap_out)

            print("\n=== Best vs random baseline (per class) ===")
            print(best_vs_baseline_df[["class_name", "mean_diff_best_minus_baseline", "wins_best", "losses_best", "p_value_adj", "significant_adj"]].head(10))
            print(f"\nSalvati: {best_vs_baseline_out.name}, {heatmap_out.name}")
        except ValueError as exc:
            print(f"\n[INFO] Skip best-vs-baseline per classe: {exc}")
    else:
        print("\n[INFO] per_class_metrics_detailed.csv non trovato in results_gat: salto analisi per-classe.")
