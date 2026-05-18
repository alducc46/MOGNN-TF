#%%
import torch
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from utils_new import downsample_data
import scipy.sparse as sp
from torch.utils.data import TensorDataset


high_variance_path = '../../datasets/data/omics/expression_variance.csv'
expression_data_path = '../../datasets/data/omics/fpkm_processed_normalized.csv'
cnv_data_path = '../../datasets/data/omics/cnv_processed.csv'
edges_file = '../../datasets/data/GGI/GGI_edges_ensembl_undirected.csv'
adj_matrix_path = '../../datasets/data/prior_knowledge/GGI_adjacency_sparse.npz'
unique_genes_path = '../../datasets/data/prior_knowledge/GGI_nodes_list.csv'
labels_path = '../../datasets/data/omics/clinical_processed.csv'
apply_filter = True


#%%
gene_gene_adj_selected, data, sample_labels, shuffle_indices = downsample_data(
    expression_data_path,
    cnv_data_path,
    labels_path,
    unique_genes_path,
    high_variance_path,
    adj_matrix_path,
    N=1000,
    apply_filter=apply_filter
)
#%%
data = data.astype(np.float32)
mean = data.mean()
std  = data.std() + 1e-6
data = (data - mean) / std

# %%
print("Data shapes after downsampling:")
print(f"Adjacency matrix: {gene_gene_adj_selected.shape}") 
print(f"Data matrix: {data.shape}")
print(f"Sample labels: {sample_labels.shape}")
print(f"Shuffle indices: {shuffle_indices.shape}")
# %%
gene_gene_adj_selected.setdiag(0)
gene_gene_adj_selected = gene_gene_adj_selected + sp.eye(gene_gene_adj_selected.shape[0])
print("Non-zero entries in adjacency matrix after adding self-loops:", gene_gene_adj_selected.nnz)
# %%
# print number of columns and rows in adjacency matrix
print(f"Adjacency matrix shape: {gene_gene_adj_selected.shape}")
# %%
# ------------------------------------------------------------
# TRAIN / VAL / TEST SPLIT
# ------------------------------------------------------------
#%%
# Calcola le dimensioni dei set basandosi sui ratio specificati
# train_ratio=0.8 -> 80% training
# test_ratio=0.1 -> 10% test, 10% validation
shuffle_index = shuffle_indices.astype(np.int32).reshape(-1)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
#train_size = int(len(shuffle_index) * train_ratio)           # 70% per training
#val_size = int(len(shuffle_index) * (1 - test_ratio))  
#print(f"Train size: {train_size}, Validation size: {val_size - train_size}, Test size: {len(shuffle_index) - val_size}")
#%%
n = len(shuffle_index)


# Calcolo posizioni di split
train_end = int(n * train_ratio)
val_end   = int(n * (train_ratio + val_ratio))
print(f"Train end index: {train_end}, Validation end index: {val_end}")

# Indici suddivisi
train_idx = shuffle_index[:train_end]
val_idx   = shuffle_index[train_end:val_end]
test_idx  = shuffle_index[val_end:]

# Applica lo split ai dati
train_data = data[train_idx]
val_data   = data[val_idx]
test_data  = data[test_idx]
train_data = train_data.astype(np.float32)
val_data   = val_data.astype(np.float32)
test_data  = test_data.astype(np.float32)

train_labels = sample_labels[train_idx]
val_labels   = sample_labels[val_idx]
test_labels  = sample_labels[test_idx]

# Converti in torch tensors
train_data = torch.FloatTensor(train_data)
val_data   = torch.FloatTensor(val_data)
test_data  = torch.FloatTensor(test_data)

train_labels = torch.LongTensor(train_labels)
val_labels   = torch.LongTensor(val_labels)
test_labels  = torch.LongTensor(test_labels)

print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")#%%
#train_data = np.asarray(data).astype(np.float32)[shuffle_index[0:train_size]]           # primi 70%
#val_data = np.asarray(data).astype(np.float32)[shuffle_index[train_size:val_size]]      # 70%-90%  
#test_data = np.asarray(data).astype(np.float32)[shuffle_index[val_size:]]               # ultimi 10%
#print(f"Train data shape: {train_data.shape}")
#print(f"Validation data shape: {val_data.shape}")
#print(f"Test data shape: {test_data.shape}")
#%%
# Divide le ETICHETTE usando gli stessi indici
#train_labels = sample_labels[np.array(shuffle_index[0:train_size])]     # etichette training
#val_labels = sample_labels[shuffle_index[train_size:val_size]]          # etichette validation
#test_labels = sample_labels[shuffle_index[val_size:]]  
#%%
# Converte i dati numpy in tensori PyTorch
#train_data = torch.FloatTensor(train_data)      # dati training in formato float32
#test_data = torch.FloatTensor(test_data)        # dati test in formato float32
#val_data = torch.FloatTensor(val_data)          # dati validation in formato float32

# Converte le etichette in LongTensor (richiesto per CrossEntropyLoss)
#train_labels = torch.LongTensor(train_labels)   # etichette training
#test_labels = torch.LongTensor(test_labels)     # etichette test  
#val_labels = torch.LongTensor(val_labels)     

#%%
# Crea dataset PyTorch che combina dati e etichette
#dset_train = TensorDataset(train_data, train_labels)
## CREAZIONE DEI DATA LOADER
# Data loader per il training: batch casuali per migliorare l'apprendimento
#train_loader = DataLoader(dset_train, batch_size=16, shuffle=True)

# Data loader per il test: nessun shuffle per risultati riproducibili
#dset_test = TensorDataset(test_data, test_labels)
#test_loader = DataLoader(dset_test, batch_size=16, shuffle=False)

# Data loader per la validation: batch casuali per valutazione robusta
#dset_val = TensorDataset(val_data, val_labels)
#val_loader = DataLoader(dset_val, batch_size=16, shuffle=True)

## SELEZIONE DEL DEVICE DI COMPUTAZIONE
# Priorità: CUDA (GPU NVIDIA) > MPS (GPU Apple) > CPU
# CUDA è preferito per le prestazioni, MPS per Mac con chip Apple Silicon
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

print('Device utilizzato per il training:', device)
#%%
edge_index, edge_weight = from_scipy_sparse_matrix(gene_gene_adj_selected)
edge_weight = edge_weight.float()  # <-- conversione fondamentale!
edge_index = edge_index.long()     # giusto per sicurezza
#%%
# ------------------------------------------------------------
# 4. COSTRUISCI UN GRAFO PER OGNI PAZIENTE
# ------------------------------------------------------------

def build_graphs(data_split, labels_split):
    graphs = []
    for i in range(len(data_split)):
        x = torch.tensor(data_split[i], dtype=torch.float32)  # (1000, 2)
        y = torch.tensor(labels_split[i], dtype=torch.long)

        g = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight.float(),
            y=y
        )
        graphs.append(g)
    return graphs



#%%
# Crea il grafo PyTorch Geometric
train_graphs = build_graphs(train_data, train_labels)
val_graphs   = build_graphs(val_data, val_labels)
test_graphs  = build_graphs(test_data, test_labels)

print("\n--- GRAPHS BUILT ---")
print("Train graphs:", len(train_graphs))
print("Validation graphs:", len(val_graphs))
print("Test graphs:", len(test_graphs))



#node_features = torch.FloatTensor(data.T)
#edge_weight = edge_attr.float()
#graph_data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
#print(f"Graph created: {graph_data.num_nodes} nodes (genes), {graph_data.num_edges} edges")
#%%
# ------------------------------------------------------------
# 5. PYTORCH GEOMETRIC DATALOADER
# ------------------------------------------------------------

train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_graphs,   batch_size=16, shuffle=False)
test_loader  = DataLoader(test_graphs,  batch_size=16, shuffle=False)
#%%
# ------------------------------------------------------------
# 6. MODELLO GCN
# ------------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nDevice:", device)

#%%
from model import SimpleGCN1
num_classes = len(np.unique(sample_labels))
#%%
modelgcn = SimpleGCN1(num_node_features=2, hidden_dim=64, num_classes=num_classes)
modelgcn = modelgcn.to(device)
print("\n--- MODELLO GCN ---")
print(modelgcn)

optimizer = torch.optim.Adam(modelgcn.parameters(), lr=1e-4, weight_decay=5e-4)
#criterion = torch.nn.NLLLoss()
criterion = CrossEntropyLoss()


#%%
# ------------------------------------------------------------
# 7. TRAINING LOOP
# ------------------------------------------------------------

print("\n--- TRAINING ---")

for epoch in range(1, 200):
    modelgcn.train()
    total_loss = 0
    train_correct = 0
    train_total = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = modelgcn(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Calcola accuratezza training
        pred = out.argmax(dim=1)
        train_correct += (pred == batch.y).sum().item()
        train_total += batch.y.size(0)

    print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")
    # === VALIDAZIONE ===
    if epoch % 5 == 0:  # Valida ogni 5 epoche
        modelgcn.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = modelgcn(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
                loss = criterion(out, batch.y)
                
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.3f}")
        print(f"          | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f}")
        print("-" * 50)
#%%
# ------------------------------------------------------------
# 8. VALUTAZIONE DEL MODELLO
# ------------------------------------------------------------
modelgcn.eval()
correct = 0
total = 0
# exporting predictions and true labels for test set
all_preds = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        out = modelgcn(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())

print("Validation accuracy:", correct/total)
# %%
# Save predictions and true labels to CSV with sample id from tcga
val_indices = shuffle_indices[train_end:val_end]
val_sample_ids = pd.read_csv(labels_path).iloc[val_indices]['sample_id'].values
val_results_df = pd.DataFrame({
    'sample_id': val_sample_ids,
    'true_label': all_labels,
    'predicted_label': all_preds
})
# %%
val_results_df.to_csv("validation_results.csv", index=False)
# %%
