#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggrega i risultati di tutti gli esperimenti in result/ e calcola media e deviazione standard.
Gestisce sia modelli GAT che GCN con diversi seed.
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path


def parse_test_metrics(filepath):
    """Parse test_metrics.txt per estrarre metriche overall."""
    if not os.path.exists(filepath):
        return None
    
    metrics = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Estrai metriche overall
    patterns = {
        'accuracy': r'accuracy:\s+([\d.]+)',
        'f1_micro': r'f1 \(micro\):\s+([\d.]+)',
        'f1_macro': r'f1_macro:\s+([\d.]+)',
        'f1_weighted': r'f1_weighted:\s+([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
    
    return metrics


def load_per_class_metrics(filepath):
    """Carica metriche per classe da CSV."""
    if not os.path.exists(filepath):
        return None
    
    return pd.read_csv(filepath)


def extract_experiment_info(folder_name):
    """Estrae informazioni dall'experimento dal nome della cartella."""
    # Pattern: {model}_{features}_seed_{seed}[_extra]
    # Es: gat_700_genes_seed_2020, gcn_700_genes_seed_2021_dropout_0.2
    
    parts = folder_name.split('_')
    
    info = {
        'model': None,
        'seed': None,
        'extra': '',
        'full_name': folder_name
    }
    
    # Estrai modello (primo elemento)
    if parts:
        info['model'] = parts[0].upper()
    
    # Estrai seed
    if 'seed' in parts:
        idx = parts.index('seed')
        if idx + 1 < len(parts):
            info['seed'] = parts[idx + 1]
            # Tutto dopo il seed è "extra"
            if idx + 2 < len(parts):
                info['extra'] = '_'.join(parts[idx + 2:])
    
    return info


def aggregate_experiments(result_dir='result'):
    """Aggrega tutti gli esperimenti."""
    
    # Trova tutte le cartelle di risultati
    result_path = Path(result_dir)
    if not result_path.exists():
        raise ValueError(f"Cartella {result_dir} non trovata")
    
    all_folders = [f for f in os.listdir(result_dir) 
                   if os.path.isdir(os.path.join(result_dir, f))]
    
    # Filtra solo cartelle con pattern seed ed escludi quelle con dropout
    seed_folders = [f for f in all_folders 
                    if 'seed' in f.lower() and 'dropout' not in f.lower()]
    
    print(f"\n=== Trovate {len(seed_folders)} cartelle con seed ===")
    
    overall_data = []
    per_class_data = []
    
    for folder in sorted(seed_folders):
        folder_path = os.path.join(result_dir, folder)
        
        # Estrai info esperimento
        exp_info = extract_experiment_info(folder)
        
        # Carica metriche overall
        metrics_file = os.path.join(folder_path, 'test_metrics.txt')
        metrics = parse_test_metrics(metrics_file)
        
        if metrics:
            metrics.update(exp_info)
            overall_data.append(metrics)
            print(f"✓ {folder}: {metrics['model']} seed={metrics['seed']}")
        else:
            print(f"✗ {folder}: test_metrics.txt non trovato o non parsabile")
            continue
        
        # Carica metriche per classe
        per_class_file = os.path.join(folder_path, 'test_per_class_metrics.csv')
        per_class_df = load_per_class_metrics(per_class_file)
        
        if per_class_df is not None:
            per_class_df['model'] = exp_info['model']
            per_class_df['seed'] = exp_info['seed']
            per_class_df['extra'] = exp_info['extra']
            per_class_df['full_name'] = exp_info['full_name']
            per_class_data.append(per_class_df)
    
    if not overall_data:
        raise ValueError("Nessun dato trovato")
    
    # Crea DataFrame
    overall_df = pd.DataFrame(overall_data)
    
    if per_class_data:
        per_class_combined = pd.concat(per_class_data, ignore_index=True)
    else:
        per_class_combined = None
    
    return overall_df, per_class_combined


def compute_statistics(df, group_cols, metric_cols):
    """Calcola statistiche per gruppo."""
    stats = df.groupby(group_cols)[metric_cols].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
    stats.columns = ['_'.join(col).rstrip('_') for col in stats.columns.values]
    return stats


def save_results(overall_df, per_class_df, output_dir='aggregated_results'):
    """Salva tutti i risultati."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva dati raw
    overall_raw_file = os.path.join(output_dir, 'all_experiments_overall.csv')
    overall_df.to_csv(overall_raw_file, index=False)
    print(f"\nSalvato: {overall_raw_file}")
    
    if per_class_df is not None:
        per_class_raw_file = os.path.join(output_dir, 'all_experiments_per_class.csv')
        per_class_df.to_csv(per_class_raw_file, index=False)
        print(f"Salvato: {per_class_raw_file}")
    
    # === STATISTICHE OVERALL PER MODELLO ===
    print("\n=== Statistiche per modello ===")
    
    metric_cols = ['accuracy', 'f1_micro', 'f1_macro', 'f1_weighted']
    
    # Raggruppa per modello (ignorando extra per ora)
    model_stats = compute_statistics(overall_df, ['model'], metric_cols)
    
    model_stats_file = os.path.join(output_dir, 'stats_by_model.csv')
    model_stats.to_csv(model_stats_file, index=False)
    print(f"Salvato: {model_stats_file}")
    
    print("\n" + model_stats.to_string(index=False))
    
    # Se ci sono configurazioni diverse (extra non vuoto), raggruppa anche per quelle
    if (overall_df['extra'] != '').any():
        config_stats = compute_statistics(overall_df, ['model', 'extra'], metric_cols)
        config_stats_file = os.path.join(output_dir, 'stats_by_config.csv')
        config_stats.to_csv(config_stats_file, index=False)
        print(f"\nSalvato: {config_stats_file}")
    
    # === STATISTICHE PER CLASSE ===
    if per_class_df is not None:
        print("\n=== Statistiche per classe ===")
        
        class_metric_cols = ['precision', 'recall', 'f1-score', 'accuracy']
        
        # Per modello e classe
        class_stats = compute_statistics(per_class_df, ['model', 'class'], class_metric_cols)
        
        class_stats_file = os.path.join(output_dir, 'stats_per_class_by_model.csv')
        class_stats.to_csv(class_stats_file, index=False)
        print(f"Salvato: {class_stats_file}")
        
        # Crea markdown per classe
        create_class_markdown(class_stats, output_dir)
    
    # === SUMMARY MARKDOWN ===
    create_summary_markdown(overall_df, model_stats, output_dir)


def create_class_markdown(class_stats, output_dir):
    """Crea file markdown con statistiche per classe."""
    
    # Separate by model
    models = class_stats['model'].unique()
    
    for model in models:
        model_data = class_stats[class_stats['model'] == model]
        
        md_file = os.path.join(output_dir, f'per_class_stats_{model.lower()}.md')
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Metriche per Classe - {model}\n\n")
            f.write("Media e Deviazione Standard su tutti i seed\n\n")
            
            f.write("| Classe | Precision | Recall | F1-Score | Accuracy | N |\n")
            f.write("|--------|-----------|--------|----------|----------|---|\n")
            
            for _, row in model_data.iterrows():
                class_id = int(row['class']) if pd.notna(row['class']) else row['class']
                f.write(f"| {class_id} | "
                       f"{row['precision_mean']:.4f}&plusmn;{row['precision_std']:.4f} | "
                       f"{row['recall_mean']:.4f}&plusmn;{row['recall_std']:.4f} | "
                       f"{row['f1-score_mean']:.4f}&plusmn;{row['f1-score_std']:.4f} | "
                       f"{row['accuracy_mean']:.4f}&plusmn;{row['accuracy_std']:.4f} | "
                       f"{int(row['precision_count'])} |\n")
        
        print(f"Salvato: {md_file}")


def create_summary_markdown(overall_df, model_stats, output_dir):
    """Crea file markdown riassuntivo."""
    
    md_file = os.path.join(output_dir, 'summary.md')
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# Summary - Risultati Aggregati\n\n")
        
        # Conta esperimenti per modello
        model_counts = overall_df.groupby('model').size()
        
        f.write("## Esperimenti Analizzati\n\n")
        f.write("| Modello | N. Esperimenti |\n")
        f.write("|---------|----------------|\n")
        for model, count in model_counts.items():
            f.write(f"| {model} | {count} |\n")
        
        f.write("\n## Metriche Overall per Modello\n\n")
        f.write("| Modello | Accuracy | F1-Macro | F1-Weighted | F1-Micro |\n")
        f.write("|---------|----------|----------|-------------|----------|\n")
        
        for _, row in model_stats.iterrows():
            f.write(f"| {row['model']} | "
                   f"{row['accuracy_mean']:.4f}&plusmn;{row['accuracy_std']:.4f} | "
                   f"{row['f1_macro_mean']:.4f}&plusmn;{row['f1_macro_std']:.4f} | "
                   f"{row['f1_weighted_mean']:.4f}&plusmn;{row['f1_weighted_std']:.4f} | "
                   f"{row['f1_micro_mean']:.4f}&plusmn;{row['f1_micro_std']:.4f} |\n")
        
        # Dettagli per seed
        f.write("\n## Dettagli per Seed\n\n")
        
        for model in overall_df['model'].unique():
            f.write(f"\n### {model}\n\n")
            f.write("| Seed | Config | Accuracy | F1-Macro | F1-Weighted |\n")
            f.write("|------|--------|----------|----------|-------------|\n")
            
            model_data = overall_df[overall_df['model'] == model].sort_values('seed')
            for _, row in model_data.iterrows():
                config = row['extra'] if row['extra'] else '-'
                f.write(f"| {row['seed']} | {config} | "
                       f"{row['accuracy']:.4f} | "
                       f"{row['f1_macro']:.4f} | "
                       f"{row['f1_weighted']:.4f} |\n")
    
    print(f"Salvato: {md_file}")


def main():
    print("="*60)
    print("AGGREGAZIONE RISULTATI ESPERIMENTI")
    print("="*60)
    
    # Aggrega esperimenti
    overall_df, per_class_df = aggregate_experiments()
    
    # Salva risultati
    save_results(overall_df, per_class_df)
    
    print("\n" + "="*60)
    print("COMPLETATO!")
    print("="*60)


if __name__ == "__main__":
    main()
