import yaml
import subprocess
import sys
import argparse
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_single_experiment(exp_config):
    """Esegue un singolo esperimento con la configurazione fornita"""
    
    # Costruisci gli argomenti per cancer_test.py
    args = [
        sys.executable, "cancer_test.py",
        "--model", str(exp_config.get('model', 'baseline')),
        "--big_lr", str(exp_config.get('big_lr', False)),
        "--num_gene", str(exp_config.get('num_gene', 700)),
        "--omic_mode", str(exp_config.get('omic_mode', 4)),
        "--batch_size", str(exp_config.get('batch_size', 64)),
        "--epochs", str(exp_config.get('epochs', 100)),
        "--dropout", str(exp_config.get('dropout', 0.6)),
        "--poolsize", str(exp_config.get('poolsize', 16)),
        "--lr", str(exp_config.get('lr', 0.001)),
        "--gene_gene", str(exp_config.get('gene_gene', True)),
        "--mirna_gene", str(exp_config.get('mirna_gene', False)),
        "--mirna_mirna", str(exp_config.get('mirna_mirna', False)),
        "--parallel", str(exp_config.get('parallel', False)),
        "--decoder", str(exp_config.get('decoder', False)),
        "--l2", str(exp_config.get('l2', False)),
        "--seed", str(exp_config.get('seed', 2020)),
        "--split_seed", str(exp_config.get('split_seed', 2020)),
        "--experiment_name", str(exp_config.get('experiment_name', exp_config.get('name', 'experiment')))
    ]
    
    # Aggiungi parametri opzionali se presenti
    if 'num_mirna' in exp_config:
        args.extend(["--num_mirna", str(exp_config['num_mirna'])])
    if 'num_tf' in exp_config:
        args.extend(["--num_tf", str(exp_config['num_tf'])])
    
    print(f"\n{'='*80}")
    print(f"Eseguendo esperimento: {exp_config.get('experiment_name', exp_config.get('name'))}")
    print(f"Modello: {exp_config.get('model')}, Seed: {exp_config.get('seed')}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(args, check=True)
    return result.returncode == 0

def run_sweep_experiments(config_path):
    """Esegue tutti gli esperimenti definiti nel file di configurazione"""
    
    cfg = load_config(config_path)
    experiments = cfg.get('experiments', [])
    
    if not experiments:
        print("Nessun esperimento trovato nel file di configurazione!")
        return
    
    print(f"Trovati {len(experiments)} esperimenti da eseguire")
    
    successi = 0
    fallimenti = 0
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Processando esperimento...")
        try:
            if run_single_experiment(exp_config):
                successi += 1
            else:
                fallimenti += 1
        except subprocess.CalledProcessError as e:
            print(f"ERRORE durante l'esecuzione dell'esperimento: {e}")
            fallimenti += 1
        except Exception as e:
            print(f"ERRORE imprevisto: {e}")
            fallimenti += 1
    
    print(f"\n{'='*80}")
    print(f"Sweep completato!")
    print(f"Successi: {successi}/{len(experiments)}")
    print(f"Fallimenti: {fallimenti}/{len(experiments)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Esegue esperimenti dal file di configurazione')
    parser.add_argument('--config', type=str, default='configs/config.yml',
                      help='Path al file di configurazione YAML')
    
    args = parser.parse_args()
    run_sweep_experiments(args.config)