#%%
import pandas as pd
from scipy.sparse import load_npz

file_path = [
    'data/cancer/adj_matrix_biogrid.npz',
    'data/cancer/biogrid_non_null.csv',
    'data/cancer/brca_shuffle_index.tsv',
    'data/cancer/brca_subtype.csv',
    'data/cancer/cnv_data_brca.tsv',
    'data/cancer/expression_data_brca.tsv',
    'data/cancer/expression_variance.tsv',
    'data/cancer/mirna_data_brca.tsv',
    'data/cancer/standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz'
]
#%%
for path in file_path:
    if path.endswith('.npz'):
        data = load_npz(path)
        print(f'Loaded sparse matrix from {path} with shape {data.shape}')
    elif path.endswith('.csv') or path.endswith('.tsv'):
        sep = ',' if path.endswith('.csv') else '\t'
        df = pd.read_csv(path, sep=sep)
        print(f'Loaded DataFrame from {path} with shape {df.shape}')

# %%
# From expression data check all columns and compare with expression variance element of the first column and biogrid non null
expression_data = pd.read_csv('data/cancer/expression_data_brca.tsv', sep='\t')
expression_variance = pd.read_csv('data/cancer/expression_variance.tsv', sep='\t')
biogrid_non_null = pd.read_csv('data/cancer/biogrid_non_null.csv')
print(f'Expression Data shape: {expression_data.shape}')
print(f'Expression Variance shape: {expression_variance.shape}')
#%%
# Check if all genes in expression variance are in expression data columns
variance_genes = set(expression_variance.iloc[:, 0])
expression_genes = set(expression_data.columns[1:])  # Exclude first column (sample IDs)
missing_genes = expression_genes - variance_genes
print(f'Number of genes in variance not in expression data: {len(missing_genes)}')
# Check if all genes in biogrid non null are in expression data columns
biogrid_genes = set(biogrid_non_null['gene'])
missing_biogrid_genes = expression_genes - biogrid_genes
print(f'Number of genes in biogrid non null not in expression data: {len(missing_biogrid_genes)}')
# %%
#adj_matrix = load_npz('../traning_data/adj_matrix_biogrid.npz')
#adj_matrix = load_npz('data/cancer/adj_matrix_biogrid.npz')
adj_matrix = load_npz('data/cancer/standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz')
#adj_matrix = load_npz('../traning_data/standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz')
print
print(f'Adjacency matrix shape: {adj_matrix.shape}')
# check values in adjacency matrix
print(f'Adjacency matrix non-zero entries: {adj_matrix.nnz}')
# %%
# check values count in adjacency matrix
import numpy as np
unique, counts = np.unique(adj_matrix.toarray(), return_counts=True)
value_counts = dict(zip(unique, counts))
print(f'Adjacency matrix value counts: {value_counts}')
# %%
# check max and min values in adjacency matrix
print(f'Adjacency matrix max value: {adj_matrix.max()}')
print(f'Adjacency matrix min value: {adj_matrix.min()}')
# %%