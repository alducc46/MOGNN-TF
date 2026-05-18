#%%
import pandas as pd
# load all files in pancancer_tcga/processed
import os
data_dir = "pancan_tcga/processed/"
omics_dir = "data/omics"
files = [
    "clinical",
    "cnv",
    "fpkm",
    "mirna"
]
#%%
#df_clinical = pd.read_parquet(os.path.join(omics_dir, files[0] + ".csv"))
#df_cnv = pd.read_parquet(os.path.join(omics_dir, files[1] + ".csv"))
#df_expression = pd.read_parquet(os.path.join(omics_dir, files[2] + ".csv"))
#df_mirna = pd.read_parquet(os.path.join(omics_dir, files[3] + ".csv"))
df_clinical = pd.read_csv(os.path.join(omics_dir, files[0] + ".csv"))
df_cnv = pd.read_csv(os.path.join(omics_dir, files[1] + ".csv"))
df_expression = pd.read_csv(os.path.join(omics_dir, files[2] + ".csv"))
df_mirna = pd.read_csv(os.path.join(omics_dir, files[3] + ".csv"))
# print the shape of each dataframe
print("Clinical shape:", df_clinical.shape)
print("CNV shape:", df_cnv.shape)
print("Expression shape:", df_expression.shape)
print("miRNA shape:", df_mirna.shape)
# print the first 5 rows of each dataframe
# %%
print(df_clinical.head())
print(df_cnv.head())
print(df_expression.head())
print(df_mirna.head())
# %%
expression_parquet_path = os.path.join(data_dir, "expression.parquet")
df_expression= pd.read_parquet(expression_parquet_path)
# see the memory usage
print(df_expression.info(memory_usage='deep'))
#%%
fpkm_csv_path = "data/omics/fpkm.csv"
df_expression = pd.read_csv(fpkm_csv_path)
print(df_expression.info(memory_usage='deep'))
# %%
# check nan values
print("NaN values in expression data:")
print(df_expression.isna().sum().sum())

#%%
def build_dtypese(file) -> dict:
    """
    Legge solo l'header del file di esempio e costruisce un dizionario dtypes:
    - 'sample_id' come string (pandas StringDtype)
    - eventuale 'cohort' come category
    - tutte le altre colonne come float32
    """
    header = pd.read_csv(file, nrows=0)
    cols = header.columns.tolist()
    print(f"Columns list: {cols}")
    dtypes = {}
    for c in cols:
        if c == "sample" or c == "icluster_cluster_assignment" or c == "sample_id":
            dtypes[c] = "string"
        elif c == "cohort":
            dtypes[c] = "category"
        else:
            dtypes[c] = "float32"
    return dtypes
#%%



# %%
# export nan to csv
nan_expression = df_expression[df_expression.isna().any(axis=1)]
print(nan_expression.info(memory_usage='deep'))
# convert dtype of float64 to float32 to reduce memory usage



print(nan_expression.info(memory_usage='deep'))


#%%
nan_expression.to_csv(os.path.join(data_dir, "nan_expression.csv"), index=False)
# %%
dtype = build_dtypese("pancan_tcga/processed/nan_expression.csv")
df_expression_nan = pd.read_csv(os.path.join(data_dir, "nan_expression.csv"), dtype=dtype)
print(df_expression_nan.info(memory_usage='deep'))
# %%
# export df_expression_nan into a low memory file compressed with gzip
df_expression_nan.to_csv(os.path.join(data_dir, "nan_expression_low_memory.csv"), index=False, compression='gzip')
# %%
from matplotlib import pyplot as plt
# plot histogram of number of NaN per gene(columns) and samples(rows)
# plot histogram of number of NaN per gene(columns)
# on x gene on y number of nan
# Se hai colonne ID, escludile dal conteggio (adatta i nomi alle tue colonne)
# 2. Colonne identificative (non vogliamo contarle nei NaN di "espressione")
id_cols = [c for c in ["sample_id", "patient_id"] if c in df_expression_nan.columns]
gene_cols = [c for c in df_expression_nan.columns if c not in id_cols]

# ============================================================
# A) NUMERO DI NaN PER GENE (X = nome gene)
# ============================================================

# Conta i NaN per ciascun gene (colonna)
nan_per_gene = df_expression_nan[gene_cols].isna().sum(axis=0)   # Series: gene -> #NaN

# Ordina e prendi i top N geni con più NaN
TOP_N_GENES = 50
nan_gene_top = nan_per_gene.sort_values(ascending=False).head(TOP_N_GENES)

plt.figure(figsize=(14, 6))
plt.bar(nan_gene_top.index, nan_gene_top.values)
plt.xlabel("Gene")
plt.ylabel("Number of NaN")
plt.title(f"Top {TOP_N_GENES} genes per number of NaN")
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

# ============================================================
# B) NUMERO DI NaN PER SAMPLE (X = sample_id)
# ============================================================

# Conta i NaN per ciascun sample (riga), solo sulle colonne dei geni
nan_per_sample = df_expression_nan[gene_cols].isna().sum(axis=1)   # Series: indice riga -> #NaN

# Se hai sample_id, usalo come indice per avere le etichette corrette
if "sample_id" in df_expression_nan.columns:
    nan_per_sample.index = df_expression_nan["sample_id"]
# Ordina e prendi i top N sample con più NaN
TOP_N_SAMPLES = 50
nan_sample_top = nan_per_sample.sort_values(ascending=False).head(TOP_N_SAMPLES)

plt.figure(figsize=(14, 6))
plt.bar(nan_sample_top.index, nan_sample_top.values)
plt.xlabel("Sample ID")
plt.ylabel("Number of NaN")
plt.title(f"Top {TOP_N_SAMPLES} samples per number of NaN")
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()
# %%
from src.multiomics_gnn.pancancer_prediction.preprocessing.preprocess_pancan_traning import high_variance_expression_gene, filter_tf_adjacency_matrix
import pandas as pd
# %%
# filter genes by high variance
expression_variance_path = "traning_data/expression_variance.tsv"
number_gene = 500
high_variance_gene_list, high_variance_gene_index = high_variance_expression_gene(expression_variance_path, number_gene)
print(f"Top {number_gene} high variance genes selected.")
print("Example genes:", high_variance_gene_list[:10])
print("Example indices:", high_variance_gene_index[:10])
#%%
tf_interactions_path = "../datasets/data/prior_knowledge/TF/TF_interactions.tsv"
# check how many unique targets are in high variance gene list
tf_edges = pd.read_csv(tf_interactions_path, sep="\t")
tf_edges["HGNC.Target"] = tf_edges["HGNC.Target"].astype(str).str.strip()
unique_targets = tf_edges["HGNC.Target"].unique().tolist()
filtered_targets = [g for g in unique_targets if g in high_variance_gene_list]
print("Total unique targets in TF interactions:", len(unique_targets))
print("Unique targets in high variance gene list:", len(filtered_targets))
# %%
# filter TF adjacency matrix by high variance genes
tf_adjacency_matrix_path = "traning_data/tf_gene_adj_global.npz"
all_tf_vocab_path = "traning_data/tf_nodes_all_in_vocab.csv"
filtered_tf_adjacency_matrix, filtered_tf_vocab = filter_tf_adjacency_matrix(
    tf_gene_matrix_path=tf_adjacency_matrix_path,
    tf_nodes_in_vocab_path=all_tf_vocab_path,
    top_gene_variance_index=high_variance_gene_index,
    n_top_tf=100,
    min_deg=1,
    max_deg=None,
    percentile=98
)
#%%
print("Filtered TF vocab size:", len(filtered_tf_vocab))
print("Example TFs:", filtered_tf_vocab[:10])
print("High variance gene list size:", len(high_variance_gene_list))
print("Example high variance genes:", high_variance_gene_list[:10])
print("Filtered TF-gene adjacency matrix shape:", filtered_tf_adjacency_matrix.shape)
# %%

assert filtered_tf_adjacency_matrix.shape[0] == len(filtered_tf_vocab), "Mismatch: righe matrice != tf_list"
assert filtered_tf_adjacency_matrix.shape[1] == len(high_variance_gene_list), "Mismatch: colonne matrice != filtered_genes"
print("Filtered TF-gene adjacency matrix shape:", filtered_tf_adjacency_matrix.shape)
import numpy as np

i = 0  # o np.random.randint(filtered_tf_adjacency_matrix.shape[0])
tf_name = filtered_tf_vocab[i]

row = filtered_tf_adjacency_matrix.getrow(i)              # sparse 1 x G
target_cols = row.indices              # indici delle colonne con 1
targets = [high_variance_gene_list[j] for j in target_cols[:20]]

print("TF:", tf_name)
print("First targets from matrix:", targets)
print("row nnz:", row.nnz)

# %%
# check if adj matrix after processing is correct by comparing with old edges
from src.multiomics_gnn.pancancer_prediction.preprocessing.preprocess_pancan_traning import down_unified_data_with_TF, load_exp_cnv_and_mirna_data
#%%
expression_data, cnv_data, mirna_data = load_exp_cnv_and_mirna_data(
    expression_data_path="traning_data/expression_data_pancan.tsv",
    cnv_data_path="traning_data/cnv_data_pancan.tsv",
    mirna_data_path="traning_data/mirna_data_top_100.tsv",)
#%%
tf_adjacency_matrix_path = "traning_data/tf_gene_adj_global.npz"
all_tf_vocab_path = "traning_data/tf_nodes_all_in_vocab.csv"
data, X_tf_expr, high_variance_gene_list = down_unified_data_with_TF(
    expression_data=expression_data,
    expression_variance_path="traning_data/expression_variance.tsv",
    tf_gene_matrix_path=tf_adjacency_matrix_path,
    tf_nodes_in_vocab_path=all_tf_vocab_path,
    omic_mode=4,
    enable_tf=True,
    number_gene=500,
    cnv_data=cnv_data,
    mirna_data=mirna_data,
    adjacency_matrix_path=None,
    mirna_to_gene_matrix_path=None,
    gene_gene=None,
    mirna_gene=None,
    mirna_mirna=None
    )
#%%
# remove all non numeric columns from expression_data and mirna_data
expression_data_new = expression_data.select_dtypes(include=['float32', 'float64'])
mirna_data_new = mirna_data.select_dtypes(include=['float32', 'float64'])


# %%
import numpy as np

X = data[:, :, 0]  # (N, 700)

# 1) ricostruisci esattamente i blocchi usati nel builder
X_gene  = expression_data_new.reindex(columns=high_variance_gene_list).to_numpy(np.float32)  # (N, 500)
X_mirna = mirna_data_new.to_numpy(np.float32)                                                # (N, 100)
X_tf    = X_tf_expr.astype(np.float32, copy=False)                                       # (N, 100)

G = X_gene.shape[1]
M = X_mirna.shape[1]
T = X_tf.shape[1]

print("Shapes:", X.shape, X_gene.shape, X_mirna.shape, X_tf.shape)
print("gene block ok :",  np.allclose(X[:, :G], X_gene))
print("mirna block ok:",  np.allclose(X[:, G:G+M], X_mirna))
print("tf block ok   :",  np.allclose(X[:, G+M:G+M+T], X_tf))

# %%
#load all adjacency matrices and print their shapes
from scipy.sparse import load_npz
tf_gene_adj_global = load_npz("traning_data/tf_gene_adj_global.npz")
gg_adj = load_npz("traning_data/adj_matrix_biogrid.npz")
mirna_gene_adj_global = load_npz("traning_data/standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz")
#%%
print("TF-gene adjacency matrix shape:", tf_gene_adj_global.shape)
print("Gene-gene adjacency matrix shape:", gg_adj.shape)
print("miRNA-gene adjacency matrix shape:", mirna_gene_adj_global.shape)
# %%
# get tf mirna 
mirna_interactions_path = "../datasets/data/prior_knowledge/miRNA/miRNA_gene_interactions.csv"
mirna_edges = pd.read_csv(mirna_interactions_path)
# get unique gene targets
unique_genes = mirna_edges["HGCN Symbol"].unique().tolist()


# %%
# load mirna gene adjacency matrix
from scipy.sparse import load_npz
mirna_gene_adj = load_npz("traning_data/standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz")
print("miRNA-gene adjacency matrix shape:", mirna_gene_adj.shape)

# %%
gene_gene_adj_path = "traning_data/adj_matrix_biogrid.npz"
mirna_gene_adj_path = "traning_data/standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz"
tf_adjacency_matrix_path = "traning_data/tf_gene_adj_global.npz"
all_tf_vocab_path = "traning_data/tf_nodes_all_in_vocab.csv"
expression_variance_path = "traning_data/expression_variance.tsv"
#%%
from src.multiomics_gnn.pancancer_prediction.preprocessing.preprocess_pancan_traning import down_unified_data_with_TF
# %%

#%%
supra_adj, data, labels = down_unified_data_with_TF(
    expression_variance_path="traning_data/expression_variance.tsv",
    expression_data=expression_data,
    cnv_data=cnv_data,
    mirna_data=mirna_data,
    omic_mode=4,
    adjacency_matrix_path=gene_gene_adj_path,
    mirna_to_gene_matrix_path=mirna_gene_adj_path,
    tf_gene_matrix_path=tf_adjacency_matrix_path,
    tf_nodes_in_vocab_path=all_tf_vocab_path,
    enable_tf=True,
    gene_gene=True,
    mirna_gene=True,
    mirna_mirna=True,
    number_gene=500,
    tf_gene=True,
    tf_mirna=True,
    tf_tf=True,
    num_mirna=100,
    num_tf=100
)
# %%
# sanity shape attesa: data (N, G+M+T, 2) e supra (G+M+T, G+M+T)
N = mirna_data.shape[0]
assert data.shape[0] == N
assert data.shape[2] == 2, f"mode4 deve avere 2 canali; trovato {data.shape}"
assert supra_adj.shape[0] == data.shape[1], "supra nodes != data nodes"

# %%
print("supra_adj:", type(supra_adj), supra_adj.shape, "nnz:", supra_adj.nnz)
print("data:", data.shape, data.dtype)
print("labels:", labels.shape, labels.min(), labels.max(), "unique:", len(set(labels)))

# %%
import numpy as np
A = supra_adj.toarray()
print("Supra symmetric?", np.allclose(A, A.T, atol=1e-6))
# %%
diag = supra_adj.diagonal()
print("diag min/max:", diag.min(), diag.max(), "nonzero diag:", (diag != 0).sum(), "/", len(diag))

# %%
import pandas as pd
# for each class in icluster_cluster_assignment calculate the top 15 genes with highest variance
expression_variance_path = "traning_data_old/expression_variance.tsv"
df_expression_variance = pd.read_csv(expression_variance_path, sep='\t', index_col=0, header=0)
df_expression = pd.read_csv("traning_data_old/expression_data_pancan.tsv", sep='\t', index_col=0, header=0)
expression_variance_df = pd.read_csv(expression_variance_path, sep='\t', index_col=0, header=0)
# %%
# anova f test
from sklearn.feature_selection import f_classif, SelectKBest
X = df_expression.select_dtypes(include=['float32', 'float64']).to_numpy()
y = df_expression['icluster_cluster_assignment'].to_numpy()
print("Feature matrix X shape:", X.shape)
print("Target vector Y shape:", y.shape)
# %%
selector = SelectKBest(score_func=f_classif, k=50)
#%%
X_new = selector.fit_transform(X, y)
selected_indices = selector.get_support(indices=True)
selected_genes = df_expression.select_dtypes(include=['float32', 'float64']).columns[selected_indices].tolist()
print("Selected genes by ANOVA F-test:", len(selected_genes))
#%%
# export selected genes to tsv
with open("selected_genes_anova_50.tsv", "w") as f:
    for gene in selected_genes:
        f.write(f"{gene}\n")
# %%

all_gene_variance_list = expression_variance_df.sort_values(by='variance', ascending=False).index.tolist()
# get top 1000 genes by variance
all_gene_variance_list = all_gene_variance_list[:500]
print("Total genes in variance file:", len(all_gene_variance_list))
# %%
# get unique icluster labels
icluster_labels = df_expression['icluster_cluster_assignment'].unique().tolist()
print("icluster labels:", icluster_labels)
#%%
high_variance_gene_list = set()
for cluster in icluster_labels:
    samples_in_cluster = df_expression[df_expression['icluster_cluster_assignment'] == cluster]
    gene_variances = samples_in_cluster.var(axis=0)
    top_genes = gene_variances.nlargest(15).index.tolist()
    high_variance_gene_list.update(top_genes)
print("Total high variance genes from all clusters:", len(high_variance_gene_list))
#%%
#check how many of these genes are in all_gene_variance_list
high_variance_gene_list_global = [gene for gene in all_gene_variance_list if gene in high_variance_gene_list]
print("High variance genes present in variance file:", len(high_variance_gene_list_global))
high_variance_gene_list_not_in_global = [gene for gene in high_variance_gene_list if gene not in all_gene_variance_list]
print("High variance genes NOT present in variance file:", len(high_variance_gene_list_not_in_global))
#%%
# export high variance gene list to tsv
with open("traning_data/high_variance_gene_list_500.tsv", "w") as f:
    for gene in high_variance_gene_list:
        f.write(f"{gene}\n")
# %%
# --------------------Calcolo expression variance
# print nans or infs in expression_data
expression_data = pd.read_csv("traning_data_old/expression_data_pancan.tsv", sep='\t', index_col=0, header=0)
#%%
# ------------------------Calcolo expression f-values for ANOVA-----------------------
from sklearn.feature_selection import f_classif
# prepare data
print("Calculating expression F-values for ANOVA...")
expression_data_anova_calc = expression_data.drop(columns=["sample"])
# get dtype of columns
print("Dtypes of expression data columns:")
print(expression_data_anova_calc.dtypes)
X = expression_data_anova_calc.select_dtypes(include=['float16','float32', 'float64']).to_numpy()
print("Feature matrix X shape:", X.shape)
Y = expression_data_anova_calc["icluster_cluster_assignment"].to_numpy()
print("Target vector Y shape:", Y.shape)
# calcola f-values
f_values, p_values = f_classif(X, Y)
# costruisci DataFrame nel formato richiesto
print("len(f_values):", len(f_values))
print("len(p_values):", len(p_values))
print("n_genes (cols):", len(expression_data_anova_calc.select_dtypes(include=['float16','float32', 'float64']).columns))

expression_fvalues_df = pd.DataFrame({
    "index": expression_data_anova_calc.select_dtypes(include=['float16','float32', 'float64']).columns.astype(str),
    "f_value": f_values,
    "p_value": p_values
})

expression_fvalues_df.to_csv(
    'expression_fvalues.tsv',
    sep="\t",
    index=False
)
# %%
# %%
adj_matrix_path = "traning_data/adj_matrix_biogrid.npz"
expression_variance_path = "traning_data/expression_fvalues.tsv"
# Step 1: Carica la matrice di adiacenza
from scipy.sparse import load_npz
adj_matrix = load_npz(adj_matrix_path)
print("Adjacency matrix shape:", adj_matrix.shape)
# Remove self-loops
adj_matrix.setdiag(0)
adj_matrix.eliminate_zeros()
print("Adjacency matrix after removing self-loops, shape:", adj_matrix.shape)
# Step 2: Carica il file di varianza dell'espressione genica
import pandas as pd
expression_fvalues_df = pd.read_csv(expression_variance_path, sep="\t")
print("Expression F-values DataFrame shape:", expression_fvalues_df.shape)
# Step 3: Costruisci gli indici generali dei geni.
gene_list = expression_fvalues_df['index'].astype(str).tolist()
gene_index_map = {gene: idx for idx, gene in enumerate(gene_list)}
print("Total genes in variance file:", len(gene_list))
# %%
# Step 4: Ordinamento dei geni in base al valore F
sorted_genes = expression_fvalues_df.sort_values(by='f_value', ascending=False)['index'].astype(str).to_dict().values()
print("Top 10 genes by F-value:", list(sorted_genes)[:10])
# %%
# Step 5: Iterative filtering
# Si selezionano i primi N*factor geni e si applica il k-core filtering
N = 700 # numero totale di nodi nella matrice di adiacenza finale
shortlist_factor = 1    # parti da N*factor
k = 1                   # k-core: 1 elimina singleton; 2 elimina anche molti deg1
step_add = 200          # quanti geni nuovi aggiungi ad ogni iterazione
#%%
import numpy as np
from scipy.sparse import csr_matrix
import networkx as nx
def k_core_filtering(adj_matrix: csr_matrix, k: int):
    """
    Applica il k-core filtering alla matrice di adiacenza.
    Ritorna la matrice filtrata e gli indici dei nodi mantenuti.
    """
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    core_nodes = list(nx.k_core(G, k=k).nodes)
    filtered_matrix = adj_matrix[core_nodes, :][:, core_nodes]
    discarded_nodes = set(range(adj_matrix.shape[0])) - set(core_nodes)
    print(f"K-core filtering (k={k}): kept {len(core_nodes)} nodes, discarded {len(discarded_nodes)} nodes")
    return filtered_matrix, core_nodes, discarded_nodes

current_factor = shortlist_factor
selected_genes = set()
discarded_genes = set()
#%%












#%%
import numpy as np
import networkx as nx

# --- indici globali ordinati per F
sorted_idx = [gene_index_map[g] for g in sorted_genes]
sorted_idx = np.array(sorted_idx, dtype=int)

# --- seed iniziale
seed_size = 100
print("Initial seed size:", seed_size)
current_set = set(sorted_idx[:seed_size].tolist())
# --- iterazione
discarded_set = set()
cursor = seed_size

print("Seed:", len(current_set), "Target N:", N)
print(len(current_set) < N and cursor < len(sorted_idx))

while len(current_set) < N and cursor < len(sorted_idx):
    print("\n--- Iteration ---")
    print("Before k-core:", len(current_set), "| discarded:", len(discarded_set))
    # 1) sottografo indotto su current_set
    idx = np.array(sorted(list(current_set)), dtype=int)
    subA = adj_matrix[idx, :][:, idx].tocsr()

    # 2) k-core (k=1 elimina singleton) + scartati
    G = nx.from_scipy_sparse_array(subA)  # se errore usa from_scipy_sparse_matrix
    coreG = nx.k_core(G, k=k)

    kept_local = set(coreG.nodes())
    removed_local = list(set(G.nodes()) - kept_local)

    # mappa local -> global
    removed_global = idx[removed_local] if len(removed_local) else np.array([], dtype=int)

    for r in removed_global.tolist():
        current_set.remove(int(r))
        discarded_set.add(int(r))

    print("After k-core:", len(current_set), "| discarded:", len(discarded_set))

    # 3) aggiungi nuovi geni (top F) a blocchi
    add = sorted_idx[cursor: min(len(sorted_idx), cursor + step_add)]
    cursor += len(add)
    for a in add.tolist():
        current_set.add(int(a))

    # 4) prova a reinserire scartati se ora hanno almeno 1 collegamento verso current_set
    cur_list = np.array(sorted(list(current_set)), dtype=int)
    restored = []
    for d in list(discarded_set):
        if adj_matrix[d, cur_list].nnz > 0:
            restored.append(d)

    for d in restored:
        discarded_set.remove(int(d))
        current_set.add(int(d))

    if len(restored) > 0:
        print("Restored:", len(restored), "| current:", len(current_set))
# --- risultato finale
final_idx = np.array(sorted(list(current_set)), dtype=int)
A_final = adj_matrix[final_idx, :][:, final_idx].tocsr()
deg = np.asarray(A_final.getnnz(axis=1)).ravel()

print("\nFINAL nodes:", len(final_idx))
print("deg==0:", int((deg==0).sum()), "| deg==1:", int((deg==1).sum()), "| deg>=2:", int((deg>=2).sum()))
print("deg stats: min", int(deg.min()), "mean", float(deg.mean()), "median", float(np.median(deg)), "max", int(deg.max()))

# %%
# plot the graph of the final adjacency matrix
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
G_final = nx.from_scipy_sparse_array(A_final)
nx.draw_spring(G_final, node_size=20, node_color='blue', edge_color='gray', alpha=0.5)
plt.title("Final Adjacency Matrix Graph after Iterative k-core Filtering")
plt.show()
# %%
# use the gene names as labels for nodes
labels = {i: gene_list[final_idx[i]] for i in range(len(final_idx))}
plt.figure(figsize=(12, 12))
G_final = nx.from_scipy_sparse_array(A_final)
nx.draw_spring(G_final, labels=labels, node_size=50, node_color='green', edge_color='gray', alpha=0.7)
plt.title("Final Adjacency Matrix Graph with Gene Labels")
plt.show()
# %%
# export final gene list to tsv
with open("final_gene_list_iterative_kcore.tsv", "w") as f:
    for i in range(len(final_idx)):
        gene_name = gene_list[final_idx[i]]
        f.write(f"{gene_name}\n")
# %%
# export with pyvis
from pyvis.network import Network
net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")
G_final = nx.from_scipy_sparse_array(A_final)
net.from_nx(G_final)
# save
net.show("final_adjacency_matrix_graph.html")

# %%