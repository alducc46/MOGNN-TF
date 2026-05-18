from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops, to_undirected
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import scipy.sparse as sp


def get_high_variance_genes(high_variance_df, N=1000):
    # Assumes dataframe with columns ['gene','variance']
    if 'gene' not in high_variance_df.columns or 'variance' not in high_variance_df.columns:
        raise ValueError("high_variance_df deve contenere colonne 'gene' e 'variance'")
    top = high_variance_df.sort_values(by='variance', ascending=False).head(N)
    genes = top['gene'].astype(str).tolist()
    print(f"Selected top {len(genes)} high variance genes (N={N}).")
    return genes

def filter_high_variance_genes(expr_df, genes_to_keep):
    # Filtra le colonne presenti nell'expr_df mantenendo solo i geni trovati
    if not isinstance(genes_to_keep, (list, pd.Series)):
        genes_to_keep = list(genes_to_keep)
    present = [g for g in genes_to_keep if g in expr_df.columns]
    missing = [g for g in genes_to_keep if g not in expr_df.columns]

    if len(missing) > 0:
        print(f"Warning: {len(missing)} genes not found in expression data (first 10 shown): {missing[:10]}")
    filtered = expr_df.loc[:, present]
    print(f"Filtered expression data to {filtered.shape[1]} genes and {filtered.shape[0]} samples.")
    return filtered


def load_expression_cnv_data(expression_data_path, cnv_data_path):
    # Carica dati di espressione genica
    expression_data = pd.read_csv(expression_data_path, index_col=0, header=0)
    
    # Carica dati CNV (Copy Number Variations)
    cnv_data = pd.read_csv(cnv_data_path, index_col=0, header=0)
    cnv_data = cnv_data.astype(int)
    cnv_data = (cnv_data - cnv_data.min().min()) / (cnv_data.max().max() - cnv_data.min().min())

    print(f"Expression data shape: {expression_data.shape}")
    print(f"CNV data shape: {cnv_data.shape}")

    return expression_data, cnv_data

## ===== FUNZIONI PER PREPROCESSING E COSTRUZIONE GRAFI =====
def downsample_data(expression_data_path, cnv_data_path, labels_path, unique_genes_path,high_variance_path,adj_matrix_path, N=1000, apply_filter=True):
    # Downsample data to target_size using shuffle_indices
    expression, cnv = load_expression_cnv_data(expression_data_path, cnv_data_path)
    print("Loaded expression and CNV data.")
    print(expression.head())

    if expression.shape[0] == cnv.shape[0]:
            print('Exp and CNV sample numbers match.')
    # filter genes by unique genes in interaction
    unique_genes = pd.read_csv(unique_genes_path)["symbol"].astype(str).tolist()
    print(f"Total unique genes in interaction network: {len(unique_genes)}")
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}

    # filter high variance genes by unique genes in interaction
    high_variance_df = pd.read_csv(high_variance_path)
    high_variance_df = high_variance_df[high_variance_df['gene'].astype(str).isin(unique_genes)]
    print(f"High variance genes after filtering by interaction network genes: {high_variance_df.shape[0]}")
    # print missing genes
    missing_genes = [g for g in unique_genes if g not in high_variance_df['gene'].astype(str).tolist()]
    print(f"Missing genes from high variance list: {len(missing_genes)} (first 10 shown): {missing_genes[:10]}")
    #export filtered high variance genes
    pd.Series(missing_genes).to_csv('missing_high_variance_genes.csv', index=False)

    if apply_filter:
        high_variance_df = pd.read_csv(high_variance_path)
        print(f"Total high variance genes before filtering: {high_variance_df}")
        # filter high variance genes by unique genes in interaction
        high_variance_df = high_variance_df[high_variance_df['gene'].astype(str).isin(unique_genes)]
        print(f"High variance genes after filtering by interaction network genes: {high_variance_df}")
        # print missing genes
        missing_genes = [g for g in unique_genes if g not in high_variance_df['gene'].astype(str).tolist()]
        print(f"Missing genes from high variance list: {len(missing_genes)} (first 10 shown): {missing_genes[:10]}")


        top_genes = get_high_variance_genes(high_variance_df, N=1000)
        high_variance_gene_index = [gene_to_idx[g] for g in top_genes if g in gene_to_idx]

        expression = filter_high_variance_genes(expression, top_genes)
        cnv = filter_high_variance_genes(cnv, top_genes)

    print(f"Final high_variance_df data shape: {high_variance_df.shape}")

    num_samples = expression.shape[0]
    expression = expression.values.reshape(expression.shape[0], -1, 1)
    data = np.array(expression).reshape(num_samples, -1 ,1)
    print(data.shape)
    cnv = np.asarray(cnv).reshape(num_samples, -1 ,1)
    print(cnv.shape)
    data = np.concatenate([data,cnv], axis=2)
    print(data.shape)

    # Controllo indici nella matrice di adiacenza
    # --------------------------------------------------
    gene_gene_adj = sp.load_npz(adj_matrix_path)
    print(f"Gene-gene adjacency matrix shape: {gene_gene_adj.shape}")

    # Converti in formato COO per leggere facilmente le coordinate
    adj_coo = gene_gene_adj.tocoo()

    # Prendi le prime 5 connessioni e verifica
    for i in range(10):
        u_idx, v_idx = adj_coo.row[i], adj_coo.col[i]
        print(f"{u_idx} ({unique_genes[u_idx]})  ↔  {v_idx} ({unique_genes[v_idx]})")
    # --------------------------------------------------

    #gene_gene_adj_mat = gene_gene_adj.todense()
    #print(gene_gene_adj_mat)
    # controlla quanti uno ci sono
    gene_gene_adj = gene_gene_adj.tocsr()

    print(f"Adjacency matrix shape: {gene_gene_adj.shape}")
    print(f"Number of edges in adjacency matrix: {gene_gene_adj.nnz // 2}")  # diviso 2 perché ogni arco è contato due volte
    
    # Normalizzazione (solo se ha senso)
    mx = gene_gene_adj.max()
    mn = gene_gene_adj.min()
    if mx != mn:
        gene_gene_adj = (gene_gene_adj - mn) / (mx - mn)

    gene_gene_adj_selected = gene_gene_adj[high_variance_gene_index,:]
    gene_gene_adj_selected = gene_gene_adj_selected[:,high_variance_gene_index]

    # Selected adjacency matrix shape: (1000, 1000)
    # Number of edges in selected adjacency matrix: 1206

    print(f"Selected adjacency matrix shape: {gene_gene_adj_selected.shape}")
    print(f"Number of edges in selected adjacency matrix: {gene_gene_adj_selected.nnz // 2}")  # diviso 2 perché ogni arco è contato due volte

    # get shuffle indices from expression data
    num_samples = expression.shape[0]
    shuffle_indices = np.random.permutation(num_samples)
    print(f"Shuffle indices: {shuffle_indices}")
    print(f"len shuffle indices: {len(shuffle_indices)}")
    print("Loading labels...")
    labels_df = pd.read_csv(labels_path)
    label_encoder = LabelEncoder()
    sample_labels = torch.tensor(label_encoder.fit_transform(labels_df['BRCA_Subtype'].values), dtype=torch.long)
    print(f"Sample labels loaded: {len(sample_labels)} samples, {len(label_encoder.classes_)} classes: {label_encoder.classes_}")
    
    
    return gene_gene_adj_selected, np.asarray(data), sample_labels.numpy(), shuffle_indices

