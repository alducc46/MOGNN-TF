# Costruzione del dataset Omico
## get_raw_data.py
- Scarica specifici dataset TCGA (miRNA, proteine, dati clinici, FPKM, CNV, metilazione) da Xena Browser.
### Output
```text
raw/
├─ annotations/
│  ├─ gencode.v36.annotation.gtf.gene.probemap
│  ├─ HM27.hg38.manifest.gencode.v36.probeMap
│  └─ HM450.hg38.manifest.gencode.v36.probeMap
├─ TCGA-BRCA/
│  └─ processed/
│     ├─ TCGA-BRCA.clinical.csv
│     ├─ TCGA-BRCA.gene-level_ascat3.csv
│     ├─ TCGA-BRCA.methylation27.csv
│     ├─ TCGA-BRCA.mirna.csv
│     ├─ TCGA-BRCA.protein.csv
│     ├─ TCGA-BRCA.star_fpkm.csv
│     ├─ TCGA-BRCA.clinical.tsv.gz
│     ├─ TCGA-BRCA.gene-level_ascat3.tsv.gz
│     ├─ TCGA-BRCA.methylation27.tsv.gz
│     ├─ TCGA-BRCA.mirna.tsv.gz
│     ├─ TCGA-BRCA.protein.tsv.gz
│     └─ TCGA-BRCA.star_fpkm.tsv.gz
├─ clinical_data.csv
└─ TCGA-BRCA.all_file_raw
```
## process_data.py
- Dati clinici: Filtraggio per scludere i campioni normal-like, standardizzazione degli id TCGA-XXXX-YYY-ZZZ
- CNV: Formato ASCAT3, filtraggio dei campioni in base a quelli presenti nei dati clinici, standardizzazione degli id, pulizia dei valori Nan (colonne con tutti Nan e N° > 30%), conversione ENSEMBL -> HGNC Symbol
- Expression data: Formato FPKM, filtraggio dei campioni in base a quelli presenti nei dati clinici, standardizzazione degli id, pulizia dei valori Nan (colonne con tutti Nan e N° > 30%), conversione ENSEMBL -> HGNC Symbol
- mirna Data: filtraggio dei campioni in base a quelli presenti nei dati clinici, standardizzazione degli id, pulizia dei valori Nan (colonne con tutti Nan e N° > 30%)
- Metilazione(da implementare): conversione delle sonde usando il file di annotazione in Geni probe -> GENCODE symbol -> HGNC symbol, pulizia dei valori Nan (colonne con tutti Nan e N° > 30%)
- (Opzionale) Conversione dei CNV da ASCAT3 a GISTIC2 
- (Opzionale) Normalizzazione dei dati:
- - CNV -> Z-Score
- - Expression -> Min-Max
- - Mirna -> Min-Max
- - Metylation -> Min-Max


Merge dei 4 file sui campioni comuni.


## boruta_feature_selection.py

Implementazione dell'algoritmo BORUTA con Random Forest per la selezione delle feature genetiche più rilevanti per la classificazione dei sottotipi di cancro.

## utils.py
- get_common_samples: Trova gli ID campione comuni tra tutti i file CSV in una directory.
- filter_files_by_common_samples: Filtra i file CSV per mantenere solo gli ID campione comuni.
- normalize_sample_ids: Normalizza gli ID campione al formato TCGA-XX-YYYY-ZZ.
- check_nan: Verifica la presenza di valori NaN nei file CSV di una directory.
- check_duplicates: Controlla la presenza di ID campione duplicati nei file CSV.
- normalize(omic_type): Applica la normalizzazione colonne numeriche di un DataFrame.
## train_test_split
Splitta il train ed il test, applica la normalizzazione sul test, estrae con la feature selection top 1000 geni per varianza ed i geni più significativi con boruta. 


# Costruzione della prior knowledge graph
## get_data.py
- get_biogrid_data(): Scarica il database BioGRID per le interazioni gene-gene umane, estrae e comprime il file in GGI/GGI.zip
- get_data(): Funzione generica per scaricare file da URL
- main(): Coordina il download di BioGRID (interazioni gene-gene) e miRDB (predizioni miRNA-target) e TFLink(TF-gene)
### Output:
- GGI/GGI.zip: Database interazioni proteina-proteina da BioGRID
- miRNA/miRDB.gz: Database predizioni targeting miRNA-gene da miRDB

## refseq2gene.py
- Converte gli ID RefSeq (NM_) del database miRDB in simboli genici utilizzando MyGene
- Filtra i dati miRDB per miRNA umani (hsa-)  
- Mappa RefSeq → simbolo genico + Entrez ID + Ensembl ID
- Output: refseq2gene_mappings.tsv (tabella di conversione per 86 ID non trovati esclusi)

## load_interaction.py
**Processamento interazioni gene-gene (BioGRID):**
- Filtra e pulisce le interazioni BioGRID rimuovendo ID non validi e self-loops
- Converte Entrez ID in simboli HGNC utilizzando tabelle di conversione
- Crea grafo non diretto rimuovendo duplicati e genera matrice di adiacenza sparsa
- Output: GGI_interactions_processed.csv, GGI_edges_ensembl_undirected.csv, GGI_adjacency_sparse.npz

**Processamento interazioni miRNA-gene:**
- Merge dei dati miRDB con mappature RefSeq per ottenere simboli genici
- Filtraggio per score (soglia 80) e rimozione duplicati
- Calcolo metapath miRNA-miRNA (connessioni indirette via geni comuni)
- Output: miRNA_gene_interactions.csv, metapath_mirna_mirna.csv, miRNA_nodes_list.csv,
miRNA-miRNA_adjacency_sparse.npz

**Processamento interazioni TF-gene:**
???

