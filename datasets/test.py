import pandas as pd

# =============================
# 1) CARICA FILE HGNC
# =============================
# Il tuo file contiene queste colonne:
# HGNC ID, Status, Approved symbol, Approved name, Ensembl gene ID, NCBI gene ID

hgnc = pd.read_csv("hgcn_mart_conversion_table.txt", sep="\t")

# rinomino per semplificare
hgnc = hgnc.rename(columns={
    "Approved symbol": "HGNC_symbol",
    "Ensembl gene ID": "ensembl"
})

# rimuovo righe senza Ensembl
hgnc = hgnc.dropna(subset=["ensembl"])

# tolgo eventuali versioni dagli Ensembl ID
hgnc["ensembl"] = hgnc["ensembl"].str.split(".").str[0]


# =============================
# 2) CARICA FILE PROBEMAP GENCODE
# =============================
gencode = pd.read_csv("data/raw/annotations/gencode.v36.annotation.gtf.gene.probemap.csv", sep="\t")

# estraggo Ensembl senza versione
gencode["ensembl"] = gencode["id"].str.split(".").str[0]

# preparo solo le colonne utili
gencode = gencode[["ensembl", "gene"]].rename(columns={"gene": "GENCODE_symbol"})


# =============================
# 3) MERGE PER LA TABELLA DI CONVERSIONE
# =============================
conversion = gencode.merge(
    hgnc[["ensembl", "HGNC_symbol"]],
    on="ensembl",
    how="left"
).drop_duplicates()

# riordino le colonne
conversion = conversion[["ensembl", "GENCODE_symbol", "HGNC_symbol"]]


# =============================
# 4) SALVA LA TABELLA
# =============================
conversion.to_csv("conversion_table_ensembl_gencode_hgnc.tsv",
                  sep="\t", index=False)

print("Tabella di conversione creata!")
print(conversion.head())





# %%
# load raw tcga pancancer
import os
import numpy as np
import pandas as pd

path = "data/raw/TCGA-PANCANCER/processed/TCGA-PANCANCER.star_fpkm.csv.gz"
tcga = pd.read_csv(path, index_col=0)
# %%
tcga.to_parquet("data/raw/TCGA-PANCANCER/processed/TCGA-PANCANCER.gene-level_ascat3.parquet")
# %%
tcga.to_pickle("data/raw/TCGA-PANCANCER/processed/TCGA-PANCANCER.star_fpkm.pkl")
# %%
# print dtype info
print(tcga.dtypes)
# %%
num_cols = tcga.select_dtypes(include=["float64", "int64"]).columns
tcga[num_cols] = tcga[num_cols].astype("float32")
# %%
print(tcga.info(memory_usage="deep"))
#%%
tcga.to_parquet(
    "data/raw/TCGA-PANCANCER/processed/TCGA-PANCANCER.star_fpkm.parquet",
    compression="zstd"    # o "snappy" se preferisci
)
#%%
path2 = "data/raw/TCGA-UCS/processed/TCGA-UCS.gene-level_ascat3.csv.gz"
# %%
tcga_ucs = pd.read_csv(path2)
# %%
# transpose
tcga_ucs = tcga_ucs.T
# %%
print(tcga_ucs.info(memory_usage="deep"))
# %%
pancan_pkl = pd.read_pickle("data/raw/TCGA-PANCANCER/processed/TCGA-PANCANCER.star_fpkm.zip", compression="zip")

# %%
print(columns_info := pancan_pkl.dtypes)
# %%
from pathlib import Path
import pandas as pd


# Root che contiene tutte le cartelle TCGA-*
ROOT = Path("data/raw")   # <-- adatta questo path al tuo caso


OMIC_PATTERNS = {
    "ascat3": "gene-level_ascat3",
    "fpkm":   "star_fpkm",
    "mirna":  "mirna",
}


def find_tcga_files(root: Path, pattern: str):
    """
    Trova tutti i file TCGA-*/processed/*{pattern}.csv.gz
    (ignora i .tsv.gz).
    """
    files = sorted(root.glob(f"TCGA-*/processed/*{pattern}.csv.gz"))
    if not files:
        raise FileNotFoundError(f"Nessun file con pattern '*{pattern}.csv.gz' trovato sotto {root}")
    return files


def build_dtypes_from_example(example_file: Path) -> dict:
    """
    Legge solo l'header del file di esempio e costruisce un dizionario dtypes:
    - 'sample_id' come string (pandas StringDtype)
    - eventuale 'cohort' come category
    - tutte le altre colonne come float32
    """
    header = pd.read_csv(example_file, nrows=0)
    cols = header.columns.tolist()

    dtypes = {}
    for c in cols:
        if c == "sample_id":
            dtypes[c] = "string"
        elif c == "cohort":
            dtypes[c] = "category"
        else:
            dtypes[c] = "float32"
    return dtypes


def load_pan_omic_fast(root: Path, omic: str) -> pd.DataFrame:
    """
    Carica TUTTI i file per una certa omica (ascat3 / fpkm / mirna),
    imponendo da subito float32 alle feature, aggiungendo 'cohort'
    e restituendo un unico DataFrame pan-cancer con colonna 'sample_id'.
    """
    if omic not in OMIC_PATTERNS:
        raise ValueError(f"Omica '{omic}' non supportata. Usa: {list(OMIC_PATTERNS)}")

    pattern = OMIC_PATTERNS[omic]
    files = find_tcga_files(root, pattern)

    # Costruisco i dtypes dal primo file
    example_file = files[0]
    dtypes = build_dtypes_from_example(example_file)
    print(f"[INFO] Dtypes stimati da {example_file.name}")

    dfs = []

    for f in files:
        # Cohort = nome della cartella, es. 'TCGA-UCS'
        cohort_name = f.parents[1].name   # .../TCGA-UCS/processed/file.csv.gz

        print(f"[INFO] Leggo {omic} da {f} (cohort={cohort_name})")
        df = pd.read_csv(
            f,
            dtype=dtypes,   # tutte le feature nascono già float32
        )

        # Controllo presenza sample_id
        if "sample_id" not in df.columns:
            raise ValueError(f"{f} non contiene 'sample_id'")

        # Aggiungo colonna cohort se manca
        if "cohort" not in df.columns:
            df["cohort"] = cohort_name
        else:
            if df["cohort"].dtype == "object":
                df["cohort"] = df["cohort"].astype("category")

        # metto sample_id + cohort all'inizio
        cols = df.columns.tolist()
        non_feat = ["sample_id", "cohort"]
        feat = [c for c in cols if c not in non_feat]
        df = df[non_feat + feat]

        dfs.append(df)

    pan_df = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Pan-{omic} shape: {pan_df.shape}")
    print(pan_df.info(memory_usage="deep"))

    return pan_df

#%%
# ==========================
# ESEMPIO USO
# ==========================

# 1) Carico pan-cancer ASCAT3
pan_ascat3 = load_pan_omic_fast(ROOT, "ascat3")
# print shape
print(pan_ascat3.shape)
#%%
pan_ascat3.to_parquet(
    "data/raw/TCGA-PANCANCER/processed/TCGA-PANCANCER.gene-level_ascat3.parquet",
    compression="zstd"
)
#%%
# 2) Carico pan-cancer FPKM
pan_fpkm = load_pan_omic_fast(ROOT, "fpkm")
print(f"FPKM shape: {pan_fpkm.shape}")
pan_fpkm.to_parquet(
    "data/raw/TCGA-PANCANCER/processed/TCGA-PANCANCER.star_fpkm.parquet",
    compression="zstd"
)
#%%
# 3) Carico pan-cancer miRNA
pan_mirna = load_pan_omic_fast(ROOT, "mirna")
print(f"miRNA shape: {pan_mirna.shape}")
pan_mirna.to_parquet(
    "data/raw/TCGA-PANCANCER/processed/TCGA-PANCANCER.mirna.parquet",
    compression="zstd"
)

# %%

# load ascat parquet
import pandas as pd
path = "data/raw/TCGA-PANCANCER/processed/TCGA-PANCANCER.gene-level_ascat3.parquet"
ascat = pd.read_parquet(path)
print(ascat.info(memory_usage="deep"))
# %%
# check sample_id duplicates
duplicate_sample_ids = ascat['sample_id'][ascat['sample_id'].duplicated()].unique()
print(f"Duplicate sample IDs in ASCAT3 data: {len(duplicate_sample_ids)} duplicates found")
#%%



#%%
# select only the sample with 01A vial
ascat_norm = ascat[ascat['sample_id'].str.contains('-01A')]
print(f"ASCAT3 shape after filtering for -01A vial: {ascat_norm.shape}")
# print all of  vial not -01A samples in a new dataframe with cohort column
ascat_not_01A = ascat[~ascat['sample_id'].str.contains('-01A')]
print(f"ASCAT3 shape for non -01A vial samples: {ascat_not_01A.shape}")
#%%
visualize_df = ascat_not_01A[['sample_id', 'cohort']]
print(visualize_df.head(20))

#%%
# check if sample in ascat_not_01A has counterpart in ascat_norm after normalization
from utils import normalize_sample_ids
normalized_ids = normalize_sample_ids(ascat_not_01A['sample_id'])
ascat_norm_ids = set(ascat_norm['sample_id'])
counterpart_ids = [sid for sid in normalized_ids if sid in ascat_norm_ids]
print(f"Number of counterpart sample IDs after normalization: {len(counterpart_ids)}")

# %%
# print first 10 duplicate sample IDs
print(duplicate_sample_ids[:10])
# %%
from utils import normalize_sample_ids
# normalize sample IDs
ascat['sample_id'] = normalize_sample_ids(ascat['sample_id'])
print(ascat['sample_id'].head())
# %%
# check again duplicates after normalization
duplicate_sample_ids = ascat['sample_id'][ascat['sample_id'].duplicated()].unique()
print(f"Duplicate sample IDs in ASCAT3 data after normalization: {len(duplicate_sample_ids)} duplicates found")

# %%
print(duplicate_sample_ids[:10])
# %%
# ----------------------------TF LINK DOWNLOAD----------------------------
import pandas as pd
conversion_table_gene = "hgcn_mart_conversion_table.zip"
path_file = "TFLink_Homo_sapiens_interactions_LS_simpleFormat_v1.0.tsv.gz"
TF_df = pd.read_csv(path_file, sep="\t")
#%%
# export first 100 row
TF_df.head(100).to_csv("TFLink_Homo_sapiens_interactions_LS_simpleFormat_v1.0_first100.tsv", sep="\t", index=False)
#%%
# ---------- 2) explode target entrez multipli + map to hgnc ----------
e = TF_df.copy()

# split "801;805;808" -> righe multiple
e["NCBI.GeneID.Target"] = (
    e["NCBI.GeneID.Target"].fillna("")
      .astype(str).str.split(";")
)
e = e.explode("NCBI.GeneID.Target")
e["NCBI.GeneID.Target"] = e["NCBI.GeneID.Target"].astype(str).str.strip()
e = e[e["NCBI.GeneID.Target"] != ""]  # elimina vuoti
# ----------- explode TF ------------------
# split "801;805;808" -> righe multiple
e["NCBI.GeneID.TF"] = (
    e["NCBI.GeneID.TF"].fillna("")
      .astype(str).str.split(";")
)
e = e.explode("NCBI.GeneID.TF")
e["NCBI.GeneID.TF"] = e["NCBI.GeneID.TF"].astype(str).str.strip()
e = e[e["NCBI.GeneID.TF"] != ""]  # elimina vuoti
#%%
# drop "-" rows in interaction_df, "-" are associated with entrez IDs in sarsCov2 and HPV studies
e = e[e["NCBI.GeneID.TF"] != "-"]
e = e[e["NCBI.GeneID.Target"] != "-"]
# convertion int64 of entrez columns
e["NCBI.GeneID.TF"] = pd.to_numeric(e["NCBI.GeneID.TF"], errors="coerce").astype("Int64")
e["NCBI.GeneID.Target"] = pd.to_numeric(e["NCBI.GeneID.Target"], errors="coerce").astype("Int64")
print("total interactions:", len(e))
#%%
# drop duplicates row
e = e.drop_duplicates(subset=["NCBI.GeneID.TF", "NCBI.GeneID.Target"])
print("total unique interactions:", len(e))
#%%
# Get unique genes in TF_df
unique_genes = pd.unique(e[["NCBI.GeneID.TF", "NCBI.GeneID.Target"]].values.ravel("K"))
print("total unique genes in interactions:", len(unique_genes))
#%%
conversion_table = pd.read_csv(conversion_table_gene, sep="\t", compression='zip')
conversion_table["NCBI gene ID"] = pd.to_numeric(conversion_table["NCBI gene ID"], errors="coerce").astype("Int64")
print("total genes in conversion table:", len(conversion_table))
#%%
# filter interaction_df to keep only genes that are in conversion table
print(e.columns.values)
conv_ids = conversion_table["NCBI gene ID"].dropna()

interaction_df = e[
    e["NCBI.GeneID.TF"].isin(conv_ids) &
    e["NCBI.GeneID.Target"].isin(conv_ids)
].copy()

print("interactions after filtering:", len(interaction_df))

#interaction_df = e[e["NCBI.GeneID.TF"].isin(conversion_table["NCBI gene ID"])]
#interaction_df = e[e["NCBI.GeneID.Target"].isin(conversion_table["NCBI gene ID"])]
print("interactions after filtering:", len(interaction_df))
#%%
# Add new columns for the Ensembl ids from conversion_table and hgcn symbol
interaction_df = interaction_df.merge(conversion_table[["NCBI gene ID", "Ensembl gene ID", "Approved symbol"]], left_on="NCBI.GeneID.TF", right_on="NCBI gene ID", how="left")
interaction_df = interaction_df.rename(columns={"Ensembl gene ID": "Ensembl Interactor A", "Approved symbol": "HGCN Symbol Interactor A"})
interaction_df = interaction_df.drop(columns=["NCBI gene ID"])

interaction_df = interaction_df.merge(conversion_table[["NCBI gene ID", "Ensembl gene ID", "Approved symbol"]], left_on="NCBI.GeneID.Target", right_on="NCBI gene ID", how="left")
interaction_df = interaction_df.rename(columns={"Ensembl gene ID": "Ensembl Interactor B", "Approved symbol": "HGCN Symbol Interactor B"})
interaction_df = interaction_df.drop(columns=["NCBI gene ID"])

# print number of nan values for each column that are more than 0
print(interaction_df.isna().sum()[interaction_df.isna().sum() > 0])
#%%
interaction_final = interaction_df[['NCBI.GeneID.TF', 'HGCN Symbol Interactor A',
                                   'NCBI.GeneID.Target', 'HGCN Symbol Interactor B', 'Name.TF', 'Name.Target']]
#%%
# rename columns
interaction_final = interaction_final.rename(columns={
    'NCBI.GeneID.TF': 'Entrez.TF',
    'HGCN Symbol Interactor A': 'HGNC.TF',
    'NCBI.GeneID.Target': 'Entrez.Target',
    'HGCN Symbol Interactor B': 'HGNC.Target'
})
# %%
df = interaction_final[["Name.TF", "Name.Target"]]
# %%
# get unique genes sorted
TF_gene_list = df["Name.TF"].unique()
TF_nodes = pd.Series(TF_gene_list).sort_values()
print(f"Total unique TF genes: {len(TF_nodes)}")
# %%
ggi_node_list = pd.read_csv("data/prior_knowledge/GGI/GGI_nodes_list.csv")
ggi_nodes = pd.Series(ggi_node_list["symbol"]).sort_values()
print(f"Total unique GGI genes: {len(ggi_nodes)}")
# %%
# check nodes in both TF and GGI
print("Nodes in both TF and GGI:")
print(TF_nodes.isin(ggi_nodes).sum())
# check nodes exclusive of TF
print("Nodes in TF but not in GGI:")
print((~TF_nodes.isin(ggi_nodes)).sum())
not_contained_in_GGI = TF_nodes[~TF_nodes.isin(ggi_nodes)]
# %%
# check if interaction final is undirected
common_genes = TF_nodes[TF_nodes.isin(ggi_nodes)]
print(f"Number of common genes in TF and Target: {len(common_genes)}")
# export interaction final.
undirected_interactions = interaction_final[
    interaction_final["Name.TF"].isin(common_genes) &
    interaction_final["Name.Target"].isin(common_genes)
]
#%%
df.to_csv("data/prior_knowledge/TF/TF_TF_interactions_undirected.tsv", sep="\t", index=False)
# %%
print("Number of unique TF", len(df["Name.TF"].unique()))
print("Number of unique Target", len(df["Name.Target"].unique()))
# %%