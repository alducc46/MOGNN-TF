## import all necessary libaraies
import json
import os
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
import sklearn.metrics

import torch
torch.manual_seed(2022)
import torch.nn as nn
import torch.nn.functional as F
from utils_li import get_all_metrics, calculate_f1

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Sampler, WeightedRandomSampler, RandomSampler, Subset
# from torch_geometric.nn import GATConv, global_mean_pool
# from torch_geometric.datasets import Planetoid
# from torch_geometric.datasets import MNISTSuperpixels

# import torch_geometric.transforms as T

# import matplotlib.pyplot as plt
from utils_li import *
from layer_model import *
from OmicDataset import OmicsGraphDataset
from Scheduler import OmicScheduler
from EarlyStopper import EarlyStopper
from debug_print import *

import gc
import wandb

def get_sampler(
            train_dataset: OmicsGraphDataset, strategy: str = "random", gamma: float = 1, seeds: int = 2022
    ) -> Sampler:
        """Return the sampler (either RandomSampler or WeightedRandomSampler)

        Args:
            train_dataset (CellDataset): Dataset for training
            strategy (str, optional): Sampling strategy. Defaults to "random" (random sampling).
                Implemented are "random", "cell", "tissue", "cell+tissue".
            gamma (float, optional): Gamma scaling factor, between 0 and 1.
                1 means total balancing, 0 means original weights. Defaults to 1.

        Raises:
            NotImplementedError: Not implemented sampler is selected

        Returns:
            Sampler: Sampler for training
        """
        if strategy.lower() == "random":
            print("Using RandomSampler")
            sampling_generator = torch.Generator().manual_seed(seeds)
            sampler = RandomSampler(train_dataset, generator=sampling_generator)
            return sampler

        if strategy.lower() == "none":
            print("No sampler used")
            sampler = None
            return sampler
        
        if strategy not in {"class"}:
            raise NotImplementedError('Unknown sampling strategy. Use "random" or "class".')
        
        if strategy.lower() == "class":
            print("Using class-based WeightedRandomSampler")
            # this solution is not accurate when a subset is used since the weights are calculated on the whole training dataset
            if isinstance(train_dataset, Subset):
                ds = train_dataset.dataset
            else:
                ds = train_dataset

            weights = ds.get_weight_pancan(gamma=gamma)

            sampling_generator = torch.Generator().manual_seed(seeds)
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(train_dataset),
                replacement=True,
                generator=sampling_generator,
            )
            print("Using WeightedRandomSampler with gamma =", gamma)
            print("Weights tensor shape:", weights.shape)
            print("Weights tensor sample values:", weights[:10])

        return sampler

def evaluate(loader, device, model, args, l2_regularization):
    model.eval()

    all_true = []
    all_pred = []

    total_samples = 0
    loss_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            B = batch.num_graphs
            N = batch.num_nodes // B
            batch_x = batch.x.view(B, N, -1)
            batch_y = batch.y
            edge_w = getattr(batch, "edge_weight", None)

            if args.decoder:
                x_reconstruct, out = model(batch_x, batch.edge_index, edge_w)
                loss_batch = model.loss(x_reconstruct, batch_x, out, batch_y, l2_regularization)
            else:
                out = model(batch_x, batch.edge_index, edge_w)
                loss_batch = model.loss(
                    batch_x.view(batch_x.size(0), -1), batch_x, out, batch_y, l2_regularization
                )
            # CALCOLO METRICHE
            # print(loss_batch)
            bs = int(batch_y.size(0))
            loss_sum += float(loss_batch.item()) * bs
            total_samples += bs

            preds = out.argmax(dim=1)
            all_pred.append(preds.detach().cpu().numpy())
            all_true.append(batch_y.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    avg_loss = loss_sum / total_samples if total_samples > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return avg_loss, acc, f1_macro

def train_epoch(loader, device, model, optimizer, args, l2_regularization):
    model.train()

    all_true, all_pred = [], []
    total_samples = 0
    loss_sum = 0.0
    count = 0

    for i, batch in enumerate(loader):
        batch = batch.to(device)

        #print("B:", batch.num_graphs,
        #"num_nodes:", batch.num_nodes,
        #"x:", tuple(batch.x.shape),
        #"edge_index:", tuple(batch.edge_index.shape),
        #"edge_min/max:", int(batch.edge_index.min()), int(batch.edge_index.max()))
        count += 1

        # Batch is a disjoint union of B identical graphs (one per patient)
        # REF — Batch increments edge_index by num_nodes via __inc__ (see Batch docs): https://pytorch-geometric.readthedocs.io/en/2.7.0/generated/torch_geometric.data.Batch.html
        B = batch.num_graphs
        N = batch.num_nodes // B  # nodes per graph
        
        assert batch.num_nodes % B == 0
        assert batch.edge_index.max().item() < batch.num_nodes, "edge_index punta a nodi non esistenti"
        assert batch.edge_index.min().item() >= 0

        batch_x = batch.x.view(B, N, -1)   # [B, N, F]
        batch_y = batch.y  # [B]

        batch_edge_index = batch.edge_index
        batch_edge_weight = getattr(batch, "edge_weight", None)

        optimizer.zero_grad()

        if args.decoder:
            x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
            # decoder output shapes (omitted per-batch)
            loss_batch = model.loss(x_reconstruct, batch_x, out, batch_y, l2_regularization)
        else:
            out = model(batch_x, batch_edge_index, batch_edge_weight)
            # output shape (omitted per-batch)
            loss_batch = model.loss(batch_x.view(batch_x.size(0), -1), batch_x, out, batch_y, l2_regularization)

        loss_batch.backward()
        # --- INIZIO DEBUG GRADIENTE ---
        # Stampiamo i gradienti solo per il primo batch di ogni epoca (o ogni tot step)
        if i == 0: 
            #print(f"\n[DEBUG GRADIENTI] Epoch {epoch+1} Batch {i}")
            total_norm = 0.0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Calcola la norma (magnitudine) del gradiente per questo layer
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    
                    # Calcola statistiche base
                    grad_mean = param.grad.data.mean().item()
                    grad_max = param.grad.data.max().item()
                    
                    # Stampa solo se il gradiente è sospettosamente piccolo o nullo
                    print(f"Layer: {name:<20} | Norm: {param_norm:.6f} | Mean: {grad_mean:.8f} | Max: {grad_max:.8f}")
            
            total_norm = total_norm ** 0.5
            print(f"--> Total Gradient Norm: {total_norm:.6f}")
            
            if total_norm < 1e-5:
                print("!!! ATTENZIONE: VANISHING GRADIENT (Il modello non impara) !!!")
        # --- FINE DEBUG GRADIENTE ---
        optimizer.step()

        bs = int(batch_y.size(0))
        loss_sum += float(loss_batch.item()) * bs
        total_samples += bs

        preds = out.argmax(dim=1)
        all_pred.append(preds.detach().cpu().numpy())
        all_true.append(batch_y.detach().cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    
    avg_loss = loss_sum / total_samples if total_samples > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return avg_loss, acc, f1_macro

def get_scheduler(optimizer, scheduler_name, total_epochs):
    scheduler = OmicScheduler(optimizer, scheduler_name, total_epochs)
    print(f"Using {scheduler.scheduler_name} scheduler.")
    return scheduler

def generate_stratified_shuffle_index(label_path, output_path_prefix):
    from sklearn.model_selection import train_test_split
    # split data into train test and validation sets with stratified sampling and save the index
    # train size 60%, val size 20%, test size 20%
    print('Loading labels from:', label_path)
    # Load labels
    labels_df = pd.read_csv(label_path)
    labels = labels_df['icluster_cluster_assignment'].values
    indices = np.arange(len(labels))
    # First split into train+val and test
    train_val_indices, test_indices, train_val_labels, test_labels = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=42)
    # Then split train+val into train and val
    train_indices, val_indices, train_labels, val_labels = train_test_split(
        train_val_indices, train_val_labels, test_size=0.25, stratify=train_val_labels, random_state=42)
    # save train test val indices into three separate tsv files
    np.savetxt(output_path_prefix + 'train_index.tsv', train_indices, fmt='%d', delimiter='\t')
    np.savetxt(output_path_prefix + 'val_index.tsv', val_indices, fmt='%d', delimiter='\t')
    np.savetxt(output_path_prefix + 'test_index.tsv', test_indices, fmt='%d', delimiter='\t')



gc.collect()

torch.cuda.empty_cache()

## set all the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default = 0.01, help='learning rate.')
parser.add_argument('--big_lr', type=str2bool, nargs='?', default = True, help='use the larger learning rate.')
parser.add_argument('--num_gene', type=int, default = 1000, help='# of genes')
parser.add_argument('--omic_mode', type=int, default = 0, help='which modes of omic to use')
parser.add_argument('--num_omic', type=int, default = 1, help='# of the omic(s) used')
parser.add_argument('--cancer_subtype', type=str2bool, nargs='?', default = False, help='if use the cancer subtype for classification')
parser.add_argument('--specific_type',type=str, default='brca', choices=['brca','luad'], help='which cancer type to use for subtype classification')
parser.add_argument('--shuffle_index',type=int, default=0, help='which shuffle index to use')
parser.add_argument('--batch_size', type=int, default = 16, help='# of genes')
parser.add_argument('--epochs', type=int, default = 100, help='# of epoch')
parser.add_argument('--dropout', type=float, default = 0.6, help='dropout rate')
parser.add_argument('--model', type=str, default = 'gat', choices=['gat','gatv2','gcn','multi-gcn','baseline'], help='which model to use')
parser.add_argument('--decay', type=float, default = 0.9, help='decay rate of the learing rate')
parser.add_argument('--poolsize', type=int, default = 8, help='the max pooling size')
parser.add_argument('--poolrate', type=float, default = 0.8, help='the pooling rate used in the self-attention pooling layer')
parser.add_argument('--gene_gene', type=str2bool, nargs='?', default = True, help='if use the Gene-gene inner connections')
parser.add_argument('--mirna_gene', type=str2bool, nargs='?', default = True, help='if use mirna-mrna connections')
parser.add_argument('--mirna_mirna', type=str2bool, nargs='?', default = True, help='include the meta-path within the mirna')
parser.add_argument('--parallel', type=str2bool, nargs='?', default = True, help='if use the parallel structure')
parser.add_argument('--l2', type=str2bool, nargs='?', default = True, help='if use the l2 regularization')
parser.add_argument('--decoder', type=str2bool, nargs='?', default = True, help='if use the decoder for the graph')
parser.add_argument('--edge_attribute', type=str2bool, nargs='?', default = False, help='if use multi-demension attributes for edges')
parser.add_argument('--edge_weight', type=str2bool, nargs='?', default = False, help='if use score as the edge weight instead of binary edges')
parser.add_argument('--train_ratio', type=float, default = 0.8, help='the ratio of the training data')
parser.add_argument('--test_ratio', type=float, default = 0.1, help='the ratio of the test data')
parser.add_argument('--sampler_strategy', type=str, default = 'random', help='the sampling strategy to use')
parser.add_argument('--output_folder', type=str, default = 'test_1', help='the output folder to save the results')
parser.add_argument('--scheduler', type=str, default = 'constant', help='the scheduler to use')



args = parser.parse_args()
# initialize wandb
run = wandb.init(
    project="tcga-pancan-gnn",   # scegli tu
    name=f"{args.model}_genes_{args.num_gene}_{args.epochs}_epochs_bs_{args.batch_size}_omicmode_{args.omic_mode}_sampler_{args.sampler_strategy}",    # nome della singola esecuzione
    config=vars(args),          # salva tutti gli argomenti
)
print('wandb initialized with run id:', run.id)

## double check corresponding num_omic and network options according to the selcted omic_mode
## mode 0: mRNA
## mode 1: miRNA
## mode 2: mRNA + miRNA
## mode 3: mRNA + CNV
## mode 4: mRNA + CNV + miRNA
args.num_omic = omic_mode_translation(args.omic_mode)
args.gene_gene, args.mirna_gene, args.mirna_mirna, num_mirna = validate_network_choice(args.omic_mode, args.gene_gene, args.mirna_gene, args.mirna_mirna)

if args.omic_mode == 1:
    args.num_gene = 0

if args.model == 'baseline':
    args.decoder = False
    args.parallel = False
## print the validated input arguments
print('Current arguments:')
print(args)

#path = 'data/cancer/'
if args.cancer_subtype:
    path = 'data/cancer/'
else:
    path = '../traning_data/'
print('Data path:', path)

result_path = os.path.join('results', 'cancer', args.output_folder)
# pulizia del path
result_path = os.path.normpath(result_path)
print('Results will be saved to:', result_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

if args.cancer_subtype:
    expression_variance_path = path + 'expression_variance.tsv'
    non_null_index_path = path + 'biogrid_non_null.csv'
    shuffle_index_path = path + 'brca_shuffle_index.tsv'
    clinical_data_path = path + 'brca_subtype.csv'
    expression_data_path = path + 'expression_data_brca.tsv'
    cnv_data_path = path + 'cnv_data_brca.tsv'
    mirna_data_path = path + 'mirna_data_brca.tsv'
    adjacency_matrix_path = path + 'adj_matrix_biogrid.npz'
    mirna_to_gene_matrix_path = path + 'standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz'
    cancer_subtype_label = pd.read_csv(clinical_data_path, sep=',', header=0)
    expression_data = pd.read_csv(expression_data_path, sep='\t', header=0)
    cancer_subtype_label = cancer_subtype_label[cancer_subtype_label['patient'].isin(expression_data['sample'].tolist())]
    print('Number of samples for selected cancer type before filtering:', len(cancer_subtype_label))
    cancer_subtype_label.to_csv(path + args.specific_type + '_subtype.csv', index=False)
    clinical_data_path = path + args.specific_type + '_subtype.csv'
else:
    expression_data_path = path + 'expression_data_pancan.tsv'
    cnv_data_path = path + 'cnv_data_pancan.tsv'
    mirna_data_path = path +'mirna_data_top_100.tsv'
    #shuffle_index_path = path + 'common_trimmed_shuffle_index_'+ str(args.shuffle_index) + '.tsv'
    # shuffle index da calcolare dopo
    clinical_data_path = path + 'molecular_subtype.csv'
    adjacency_matrix_path = path + 'adj_matrix_biogrid.npz'
    mirna_to_gene_matrix_path = path + 'standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz'
    expression_variance_path = path + 'expression_variance.tsv'
    non_null_index_path = path + 'biogrid_non_null.csv'
    # shuffle index calculation with stratified sampling


if not os.path.exists(path + 'common_trimmed_shuffle_index_train_index.tsv'):
    print('Generating shuffle index with stratified sampling...')
    generate_stratified_shuffle_index(clinical_data_path,
                                    path + 'common_trimmed_shuffle_index_')

train_shuffle_index = pd.read_csv(path + 'common_trimmed_shuffle_index_train_index.tsv', sep='\t', header=None).values.flatten()
val_shuffle_index = pd.read_csv(path + 'common_trimmed_shuffle_index_val_index.tsv', sep='\t', header=None).values.flatten()
test_shuffle_index = pd.read_csv(path + 'common_trimmed_shuffle_index_test_index.tsv', sep='\t', header=None).values.flatten()

## use the loading function to load the data
if args.omic_mode < 3:
    expr_all_data, mirna_all_data = load_exp_and_real_mirna_data(expression_data_path, mirna_data_path)

    adj, train_data_all, labels = down_sampling_exp_and_real_mirna_data(expression_variance_path=expression_variance_path,
                                                                        expression_data=expr_all_data,
                                                                        mirna_data=mirna_all_data,
                                                                        omic_mode=args.omic_mode,
                                                                        non_null_index_path=non_null_index_path,
                                                                        adjacency_matrix_path=adjacency_matrix_path,
                                                                        mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,
                                                                        gene_gene=args.gene_gene,
                                                                        mirna_gene=args.mirna_gene,
                                                                        mirna_mirna=args.mirna_mirna,
                                                                        number_gene=args.num_gene,
                                                                        singleton=False)
else:
    expr_all_data, cnv_all_data, mirna_all_data = load_exp_cnv_and_real_mirna_data(expression_data_path, cnv_data_path, mirna_data_path)

    adj, train_data_all, labels = down_sampling_exp_cnv_and_real_mirna_data(expression_variance_path=expression_variance_path,
                                                                        expression_data=expr_all_data,
                                                                        cnv_data=cnv_all_data,
                                                                        mirna_data=mirna_all_data,
                                                                        omic_mode=args.omic_mode,
                                                                        non_null_index_path=non_null_index_path,
                                                                        adjacency_matrix_path=adjacency_matrix_path,
                                                                        mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,
                                                                        gene_gene=args.gene_gene,
                                                                        mirna_gene=args.mirna_gene,
                                                                        mirna_mirna=args.mirna_mirna,
                                                                        number_gene=args.num_gene,
                                                                        singleton=False)

#-----DEBUG PRINTS SUPRA_ADJ -----
debug_sparse_matrix("SUPRA_ADJ (raw from down_sampling)", adj)
# hardcoded 100 (come mi hai chiesto): 700 geni + 100 miRNA = 800
debug_supra_blocks(adj, num_gene=args.num_gene, num_mirna=100)
## filter data if a specific cancer type is selected

if args.cancer_subtype:
    print('Filtering data for specific cancer type:', args.specific_type)
    train_data_all, labels = filter_data_by_cancer_type(clinical_data_path,
                                                        train_data_all,
                                                        expr_all_data)
print('Total number of samples after filtering:', len(labels))
print(train_data_all)
print(labels)

## process the labels to make sure it starts from 0
from sklearn import preprocessing
#from torch_sparse import SparseTensor
print('Labels before encoding:', np.unique(labels, return_counts=True))
le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)  
print('Labels after encoding:', np.unique(labels, return_counts=True))
# if not args.singleton:      
#     adj, train_data_all = removeZeroAdj(adj, train_data_all)

## adds the self connecting edges and convert the adj to edge_index format
# ????
# Ci sono valori estremamente alti nella matrice di adiacenza
# Adjacency matrix max value: 302.0
# Adjacency matrix min value: 0.0
# ????
print('Adjacency matrix max value non zero:', np.max(adj.data))
print('Adjacency matrix min value non zero:', np.min(adj.data))
print("Adj matrix max value:", np.max(adj))
print("Adj matrix min value:", np.min(adj))

#-----DEBUG PRINTS SUPRA_ADJ -----
debug_sparse_matrix("SUPRA_ADJ (pre-normalization)", adj)

adj_for_loss = adj.todense()
adj = adj/np.max(adj)
adj = adj.astype('float32')
debug_sparse_matrix("SUPRA_ADJ (post-normalization)", adj)
adj.setdiag(0)
debug_sparse_matrix("SUPRA_ADJ (post-diag-zero)", adj)
adj = adj + sp.eye(adj.shape[0])
debug_sparse_matrix("SUPRA_ADJ (post-self-loop)", adj)

adj = sp.coo_matrix(adj)

print(f"[DEBUG] COO nnz = {adj.nnz} (questo deve coincidere con edge_index.size(1))")

G = args.num_gene      # 700
M = 100                # lascia pure hardcoded
A = adj.tocsr()

print("nnz gene-gene:", A[:G,:G].nnz)
print("nnz gene->mirna:", A[:G,G:].nnz)
print("nnz mirna->gene:", A[G:,:G].nnz)
print("nnz mirna-mirna:", A[G:,G:].nnz)
print("diag nnz:", A.diagonal().astype(bool).sum())

#edge_index = torch.stack([torch.tensor(adj.row), torch.tensor(adj.col)], dim=0)
edge_index = torch.tensor(
    np.vstack((adj.row, adj.col)),
    dtype=torch.long
)

#edge_weight = torch.Tensor(adj.data)
edge_weight = torch.tensor(adj.data, dtype=torch.float32)
# stampa numero di nodi ed edges
print('Number of nodes:', adj.shape[0])
print('Number of edges:', edge_index.size(1))
debug_edge_index(
    "EDGE_INDEX / EDGE_WEIGHT (from COO)",
    edge_index=edge_index,
    num_nodes=adj.shape[0],
    edge_weight=edge_weight
)
# Se ti serve verificare i conteggi attesi:
# NB: se hai aggiunto +I sopra, qui ti aspetti E = (offdiag_nnz + diag_nnz)
print(f"[DEBUG] adj.nnz={adj.nnz}  edge_index.E={edge_index.size(1)}  (devono essere uguali)")
## convert uniformly edge weight into multi-dimension edge attributes if needed.
if args.edge_attribute:

    edge_attribute = disassemble_edge_weights(edge_weight, edge_index, args.num_gene, args.num_omic)
    print('Edge attribute shape:', edge_attribute.shape)
    print('Edge attibute values:', edge_attribute)

print('Edge index shape:', edge_index.shape)
print('Edge weight shape:', edge_weight.shape)
print('Eghe weight values:', edge_weight)
print('Edge weight min:', edge_weight.min().item())
print('Edge weight max:', edge_weight.max().item())

print('Number of nodes:', adj.shape[0])
print('Number of edges:', edge_index.size(1))
print('Number of features per node:', train_data_all[0].shape[1])


train_data = np.asarray(train_data_all).astype(np.float32)[train_shuffle_index]
val_data = np.asarray(train_data_all).astype(np.float32)[val_shuffle_index]
test_data = np.asarray(train_data_all).astype(np.float32)[test_shuffle_index]
train_labels = labels[train_shuffle_index]
val_labels = labels[val_shuffle_index]
test_labels = labels[test_shuffle_index]
print('Train data shape:', train_data.shape)
print('Validation data shape:', val_data.shape)
print('Test data shape:', test_data.shape)

print('Train labels distribution:', np.unique(train_labels, return_counts=True))
print('Validation labels distribution:', np.unique(val_labels, return_counts=True))
print('Test labels distribution:', np.unique(test_labels, return_counts=True))

train_size = train_data.shape[0]
val_size = val_data.shape[0]
test_size = test_data.shape[0]

print('Train labels distribution:', np.unique(train_labels, return_counts=True))


# ## dropout some training samples
# train_data, train_labels = dropout_data(train_data, train_labels, 0.75)

# ll, cnt = np.unique(train_labels,return_counts=True)

nclass = len(np.unique(labels))
# print(nclass)

train_labels = train_labels.astype(np.int64)
test_labels = test_labels.astype(np.int64)
val_labels = val_labels.astype(np.int64)
train_data = torch.FloatTensor(train_data)
test_data = torch.FloatTensor(test_data)
val_data = torch.FloatTensor(val_data)
train_labels = torch.LongTensor(train_labels)
test_labels = torch.LongTensor(test_labels)
val_labels = torch.LongTensor(val_labels)

# --- PyG-native batching: one Data object per patient (same graph topology, different node features) ---
edge_feat = None
if args.edge_attribute:
    edge_feat = edge_attribute
elif args.edge_weight:
    edge_feat = edge_weight

train_ds = OmicsGraphDataset(train_data, train_labels, edge_index, edge_weight=edge_feat,edge_attr=None, transform=None, pre_transform=None)
val_ds = OmicsGraphDataset(val_data, val_labels, edge_index, edge_weight=edge_feat, edge_attr=None, transform=None, pre_transform=None)
test_ds = OmicsGraphDataset(test_data, test_labels, edge_index, edge_weight=edge_feat, edge_attr=None, transform=None, pre_transform=None)

num_nodes = train_data.shape[1]  # 800
print(edge_index.min().item())

print(edge_index.max().item())

# Debug prints to inspect the dataset class behavior
print('=== Dataset debug info ===')
print('Train dataset length:', len(train_ds))
print('Val dataset length:', len(val_ds))
print('Test dataset length:', len(test_ds))

if len(train_ds) > 0:
    sample = train_ds[0]
    print('Sample keys:', list(sample.keys()) if hasattr(sample, 'keys') else 'N/A')
    print('First train sample x shape:', getattr(sample, 'x').shape if hasattr(sample, 'x') else None)
    print('First train sample y:', getattr(sample, 'y'))
    print('First train sample edge_index shape:', getattr(sample, 'edge_index').size() if hasattr(sample, 'edge_index') else None)
    ew = getattr(sample, 'edge_weight', None)
    if ew is not None:
        try:
            print('First train sample edge_weight shape:', ew.shape)
        except Exception:
            print('First train sample edge_weight type:', type(ew))
    else:
        print('No edge_weight on sample.')
    print('Sample num_nodes:', getattr(sample, 'num_nodes', None))
    print('Sample x dtype:', getattr(sample, 'x').dtype if hasattr(sample, 'x') else None)
else:
    print('Train dataset is empty.')

# check Weight creation for WeightedRandomSampler
weights = train_ds.get_weight_pancan(gamma=1)
print('Weights tensor shape:', weights.shape)
print('Weights tensor sample values:', weights[:10])

sampler_state = args.sampler_strategy  # 'weighted' or 'random'
training_sampler = get_sampler(train_ds, strategy=sampler_state, gamma=1.0, seeds=2022)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=training_sampler)
#train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
print('Number of training batches:', len(train_loader))
print('Number of validation batches:', len(val_loader))
print('Number of test batches:', len(test_loader))
print('Dataloader traning batch size:', train_loader.batch_size)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

print('Device:', device)
#exit()
if args.model == 'gcn':
    model = GCN(args.model, 
                args.parallel, 
                args.l2, 
                args.decoder, 
                args.poolsize, 
                args.poolrate,
                args.edge_weight, 
                args.edge_attribute, 
                args.num_gene,
                num_mirna, 
                args.omic_mode, 
                nclass, 
                args.dropout).to(device)
else:
    model = GAT(args.model, 
                args.parallel, 
                args.l2, args.decoder, 
                args.poolsize, 
                args.poolrate,
                args.edge_weight, 
                args.edge_attribute, 
                args.num_gene,
                num_mirna, 
                args.omic_mode, 
                nclass, 
                args.dropout).to(device)
if args.model == 'baseline':
    model = Baseline(args.model, 
                args.parallel, 
                args.l2, args.decoder, 
                args.poolsize, 
                args.poolrate,
                args.edge_weight, 
                args.edge_attribute, 
                args.num_gene,
                num_mirna, 
                args.omic_mode, 
                nclass, 
                args.dropout).to(device)

## optimizer with adjusted lr
global_lr = args.lr
global_step = 0
decay = args.decay
decay_steps = train_size

#def adjust_learning_rate(optimizer, epoch):
#    """Sets the learning rate to the initial LR decayed by 20% every 12 epochs"""
#    print('Adjusting learning rate...')
#    if args.big_lr:
#        print('Using big learning rate schedule.')
#        print('Global step:', global_step)
#        lr = args.lr * pow(decay, float(global_step// decay_steps))
#        print(f'Adjusted learning rate: {lr:.6f}')
#    else:
#        lr = args.lr * (0.8 ** (epoch // 12))
    
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr
#    return lr

# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr= args.lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = get_scheduler(optimizer, args.scheduler, args.epochs)

l2_regularization = 5e-4

t_total_train = time.time()

train_loss_hist, val_loss_hist = [], []
train_acc_hist, val_acc_hist = [], []
train_f1_hist, val_f1_hist = [], []
lr_hist = []

early_stopper = EarlyStopper(patience=3, min_delta=0.01)

for epoch in range(args.epochs):
    print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
    t_start = time.time()

    train_loss, train_acc, train_f1 = train_epoch(
        loader=train_loader,
        device=device,
        model=model,
        optimizer=optimizer,
        args=args,
        l2_regularization=args.l2
    )
    print(f"Training set: Average loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}", f", F1 macro: {train_f1:.4f}")

    val_loss, val_acc, val_f1 = evaluate(
        loader=val_loader,
        device=device,
        model=model,
        args=args,
        l2_regularization=args.l2
    )
    print(f"Validation set: Average loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}", f", F1 macro: {val_f1:.4f}")

    # end epoch summaries
    t_stop = time.time() - t_start
    print(f"Epoch {epoch+1} completed in {t_stop:.2f} seconds.")
    # step the scheduler
    scheduler.step()
    cur_lr = scheduler.get_last_lr()[0]
    print(f"Epoca {epoch+1} completed, nuovo learning rate: {cur_lr:.6f}")

    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)
    train_f1_hist.append(train_f1)
    val_f1_hist.append(val_f1)

    lr_hist.append(cur_lr)
    if early_stopper.early_stop(val_loss) and epoch > 100:
        print("Early stopping triggered. Stopping training.")          
        break
    run.log({
    "epoch": epoch + 1,
    "lr": cur_lr,
    "train/loss": train_loss,
    "train/acc": train_acc,
    "val/loss": val_loss,
    "val/acc": val_acc,
    "train/f1": train_f1,
    "val/f1": val_f1,
    })

#test function modificata

def test(loader, unused_labels, num_classes):
    model.eval()

    true_labels = []
    pred_labels = []

    print(len(loader.dataset))

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            B = batch.num_graphs
            N = batch.num_nodes // B
            batch_x = batch.x.view(B, N, -1)
            batch_y = batch.y
            edge_w = getattr(batch, "edge_weight", None)

            if args.decoder:
                _, out = model(batch_x, batch.edge_index, edge_w)
            else:
                out = model(batch_x, batch.edge_index, edge_w)

            pred = out.argmax(dim=1)

            true_labels.append(batch_y.detach().cpu().numpy())
            pred_labels.append(pred.detach().cpu().numpy())

    true_labels = np.concatenate(true_labels) if len(true_labels) else np.array([], dtype=int)
    pred_labels = np.concatenate(pred_labels) if len(pred_labels) else np.array([], dtype=int)

    acc = sklearn.metrics.accuracy_score(true_labels, pred_labels)
    f1 = sklearn.metrics.f1_score(true_labels, pred_labels, average='macro', zero_division=0)

    global_metrics, per_class_metrics, confusion_mat, classification_report = get_all_metrics(
        true_labels, pred_labels, num_classes
    )

    class_names = [
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
        "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20",
        "C21", "C22", "C23", "C25", "C26", "C27", "C28"
    ]

    save_metrics_pretty_txt(
        global_metrics,
        per_class_metrics,
        classification_report,
        os.path.join(result_path, 'test_metrics.txt'),
        num_classes,
        class_names=class_names
    )
    save_confusion_matrix_pretty(confusion_mat, result_path, class_names)
    return acc, f1


print("Test set performance:")
test_acc, test_f1 = test(test_loader, test_labels, nclass)
print(f"test_accuracy: {test_acc:.4f}")
print(f"test_f1_score: {test_f1:.4f}")
print(f"Epoche totali eseguite: {epoch + 1}")

# ---- salva curve ----

curves_df = pd.DataFrame({
    "epoch": np.arange(1, len(train_loss_hist) + 1),
    "lr": lr_hist,
    "train_loss": train_loss_hist,
    "val_loss": val_loss_hist,
    "train_acc": train_acc_hist,
    "val_acc": val_acc_hist,
    "train_f1": train_f1_hist,
    "val_f1": val_f1_hist
})
wandb.finish()
# save learning curves to csv
curves_df.to_csv(os.path.join(result_path, "learning_curves.csv"), index=False)
# save plots of learning curves
plt.figure()
plt.plot(curves_df["epoch"], curves_df["train_loss"], label="Train Loss")
plt.plot(curves_df["epoch"], curves_df["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning Curve - Loss")
plt.legend()
plt.savefig(os.path.join(result_path, "learning_curve_loss.png"))
print("Saved learning curves to:", os.path.join(result_path, "learning_curves.csv"))
#print(test(test_loader,nclass))
# F1 score
plt.figure()
plt.plot(curves_df["epoch"], curves_df["train_f1"], label="Train F1 Score")
plt.plot(curves_df["epoch"], curves_df["val_f1"], label="Validation F1 Score")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Learning Curve - F1 Score")
plt.legend()
plt.savefig(os.path.join(result_path, "learning_curve_f1.png"))
print("Saved learning curve F1 plot to:", os.path.join(result_path, "learning_curve_f1.png"))
