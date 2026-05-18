#%%
from torch_geometric.utils import from_scipy_sparse_matrix, add_self_loops, to_undirected
from torch_geometric.data import Data
import pandas as pd

high_variance_path = 'datasets/data/omics/expression_variance.csv'
expression_data_path = 'datasets/data/omics/fpkm_processed_normalized.csv'
cnv_data_path = 'datasets/data/omics/cnv_processed_normalized.csv'
adj_matrix_path = 'datasets/data/GGI/GGI_edges_ensembl_undirected.csv'

def get_high_variance_genes(high_variance_gene_index, N=1000):
    # Ordina in base alla varianza
    high_variance_gene_index = high_variance_gene_index.sort_values(by='variance', ascending=False).head(N)
    print(f"Selected {N} high variance genes.")
    print(high_variance_gene_index)
    return high_variance_gene_index


def load_data(expression_data_path, cnv_data_path, adj_matrix_path, high_variance_gene_index_path):
    # Carica i geni ad alta varianza
    high_variance_gene_index = pd.read_csv(high_variance_gene_index_path, header=None).squeeze().tolist()
    # Carica i dati di espressione
    expression_data = pd.read_csv(expression_data_path, index_col=0)
    print(f"Expression data shape: {expression_data.shape}")
    expression_data = filter_high_variance_genes(expression_data, high_variance_gene_index)

    # Carica i dati CNV
    cnv_data = pd.read_csv(cnv_data, index_col=0)
    print(f"CNV data shape: {cnv_data.shape}")

    # Carica la matrice di adiacenza
    edges_df = pd.read_csv(adj_matrix)
    print(f"Adjacency edges shape: {edges_df.shape}")

    return expression_data, cnv_data, edges_df

#%%
df = pd.read_csv(high_variance_path)
a = get_high_variance_genes(df, N=5000)
# %%
gene_protein_coding = pd.read_csv('Tests/New_GNN/enrichment_results/high_variance_genes_protein_coding.csv')


#%%
high_variance_genes_from_multimodal = pd.read_csv('high_variance_genes.csv')
#%%
# gene enrichment su a [gene]
# Importa il modulo enrichr e dotplot da gseapy
from gseapy import enrichr, dotplot

# Lista dei geni da analizzare
#gene_list = a['gene'].tolist()  # Prendi i valori dalla colonna 'gene'
# Esegui l'enrichment analysis
#gene_list = gene_protein_coding['gene'].tolist()
gene_list = high_variance_genes_from_multimodal['genes'].tolist()
enrichment_results = enrichr(
    gene_list=gene_list,
    gene_sets='GO_Biological_Process_2023',  # Latest GO BP library
    organism='human',  # or 'mouse' if applicable
    cutoff=0.05,  # Adjusted p-value cutoff
    format='png'  # Generate plots
    #outdir='Tests/New_GNN/enrichment_results',  # Directory di output per i risultati
    #no_plot=True  # Evita di generare automaticamente i grafici
)

print("\nTop 10 enriched GO Biological Processes:")
print(enrichment_results.results.head(10))

enrichment_df = enrichment_results.results

# Visualizza i primi risultati
print(enrichment_df.head())

# Filter significant results
significant_terms = enrichment_df[enrichment_df['Adjusted P-value'] < 0.05]
print(f"\nNumber of significant GO terms (FDR < 0.05): {len(significant_terms)}")

# Save results to file
enrichment_df.to_csv('enrichment_results/go_bp_enrichment_results.csv', index=False)

# Genera un dotplot per visualizzare i risultati
dotplot(
    enrichment_df,
    title='Gene Enrichment Analysis',
    figsize=(10, 6),
    show=False,  # Non mostrare il grafico interattivamente
    ofname='enrichment_results/dotplot.png'  # Percorso e nome del file di output
)
#%%




# %%
# esporta i N geni ad alta varianza
a.to_csv('Tests/New_GNN/enrichment_results/high_variance_genes.csv', index=False)
# %%
import pandas as pd
from mygene import MyGeneInfo

# === 1. Carica i tuoi geni
gene_df = pd.read_csv("Tests/New_GNN/enrichment_results/high_variance_genes.csv")  # Deve contenere una colonna 'gene'
gene_list = gene_df['gene'].tolist()

# === 2. Interroga MyGene per recuperare tipo di gene
mg = MyGeneInfo()
results = mg.querymany(gene_list, scopes='symbol', fields='type_of_gene', species='human')

# === 3. Filtra solo quelli protein-coding
protein_coding_genes = [
    r['query'] for r in results
    if r.get('type_of_gene') == 'protein-coding' and not r.get('notfound', False)
]

# === 4. Salva su CSV
filtered_df = pd.DataFrame(protein_coding_genes, columns=['gene'])
filtered_df.to_csv("Tests/New_GNN/enrichment_results/high_variance_genes_protein_coding.csv", index=False)

# === 5. Riepilogo
print(f"Totale geni iniziali: {len(gene_list)}")
print(f"Geni protein-coding trovati: {len(protein_coding_genes)}")
print("Salvato in 'high_variance_genes_protein_coding.csv'")

# %%

import pandas as pd

# === Percorsi file ===
gene_list_path = "Tests/New_GNN/enrichment_results/high_variance_genes.csv"
enrichr_results_path = "Tests/New_GNN/enrichment_results/GO_Biological_Process_2021.Human.enrichr.reports.txt"

# === 1. Carica la lista dei tuoi geni ad alta varianza ===
high_var_genes = pd.read_csv(gene_list_path)
gene_list = set(high_var_genes['gene'])

# === 2. Carica i risultati Enrichr ===
enrichr_results = pd.read_csv(enrichr_results_path, sep='\t')
enriched_genes = set()

# Colonna 'Genes' contiene liste separate da ';'
for genes in enrichr_results['Genes']:
    enriched_genes.update(genes.split(';'))

# === 3. Geni che non compaiono in nessun termine arricchito ===
not_enriched_genes = sorted(gene_list - enriched_genes)
print(f"Totale geni ignorati: {len(not_enriched_genes)}")

# Salva su CSV (facoltativo)
pd.DataFrame(not_enriched_genes, columns=['gene']).to_csv("Tests/New_GNN/enrichment_results/ignored_genes.csv", index=False)




# %%
