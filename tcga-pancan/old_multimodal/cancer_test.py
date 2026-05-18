    ## import all necessary libaraies
import os
import time
import logging
import sys
import wandb


import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
import sklearn.metrics

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score

# from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GATConv, global_mean_pool
# from torch_geometric.datasets import Planetoid
# from torch_geometric.datasets import MNISTSuperpixels

# import torch_geometric.transforms as T

import matplotlib.pyplot as plt
from utils import *
from layer_model import *

import gc

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
parser.add_argument('--seed', type=int, default=42, help='random seed for shuffling data')
parser.add_argument('--split_seed', type=int, default=2020, help='random seed for shuffling data')
parser.add_argument('--experiment_name', type=str, default='experiment', help='name of the experiment for logging and saving results')

args = parser.parse_args()

torch.manual_seed(args.seed)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler('gradient_debug.log'),
        logging.StreamHandler(sys.stdout)
    ],
    format='%(asctime)s - %(message)s', 
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)
logger.info("Starting gradient debugging log")
print("logger set up. Logging to 'gradient_debug.log'")
run = wandb.init(
    project="li_24",   # scegli tu
    name=f"{args.model}_genes_{args.num_gene}_{args.epochs}_epochs_bs_{args.batch_size}_omicmode_{args.omic_mode}",    # nome della singola esecuzione
    config=vars(args),          # salva tutti gli argomenti
)

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
logger.info(f'Current arguments: {args}')

if args.cancer_subtype:
    print('Classifying cancer subtypes for', args.specific_type)
    path = 'data/cancer/'
else:
    print('Classifying pan-cancer types')
    path = '../traning_data/'

expression_variance_path = path + 'expression_variance.tsv'
non_null_index_path = path + 'biogrid_non_null.csv'
if args.cancer_subtype:
    if args.specific_type.lower() == 'brca':
        shuffle_index_path = path + 'brca_shuffle_index.tsv'
        cancer_subtype_label_path = path + 'brca_subtype.csv'
        expression_data_path = path + 'expression_data_brca.tsv'
        cnv_data_path = path + 'cnv_data_brca.tsv'
        mirna_data_path = path +'mirna_data_brca.tsv'
else:
    expression_data_path = path + 'expression_data_pancan.tsv'
    cnv_data_path = path + 'cnv_data_pancan.tsv'
    mirna_data_path = path +'mirna_data_top_100.tsv'
    shuffle_index_path = None
adjacency_matrix_path = path + 'adj_matrix_biogrid.npz'
mirna_to_gene_matrix_path = path + 'standardized_mirna_mrna_edge_filtered_at_eight_with_top_100_mirna.npz'
result_path = 'result/' + f"{args.experiment_name}/"
os.makedirs(result_path, exist_ok=True)

## use the loading function to load the data
if args.omic_mode < 3:
    expr_all_data, mirna_all_data = load_exp_and_real_mirna_data(expression_data_path, mirna_data_path)

    adj, train_data_all, labels, shuffle_index = down_sampling_exp_and_real_mirna_data(expression_variance_path=expression_variance_path,
                                                                        expression_data=expr_all_data,
                                                                        mirna_data=mirna_all_data,
                                                                        omic_mode=args.omic_mode,
                                                                        non_null_index_path=non_null_index_path,
                                                                        shuffle_index_path=shuffle_index_path,
                                                                        adjacency_matrix_path=adjacency_matrix_path,
                                                                        mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,
                                                                        gene_gene=args.gene_gene,
                                                                        mirna_gene=args.mirna_gene,
                                                                        mirna_mirna=args.mirna_mirna,
                                                                        number_gene=args.num_gene,
                                                                        singleton=False)
else:
    expr_all_data, cnv_all_data, mirna_all_data = load_exp_cnv_and_real_mirna_data(expression_data_path, cnv_data_path, mirna_data_path)

    adj, train_data_all, labels, shuffle_index = down_sampling_exp_cnv_and_real_mirna_data(expression_variance_path=expression_variance_path,
                                                                        expression_data=expr_all_data,
                                                                        cnv_data=cnv_all_data,
                                                                        mirna_data=mirna_all_data,
                                                                        omic_mode=args.omic_mode,
                                                                        non_null_index_path=non_null_index_path,
                                                                        shuffle_index_path=shuffle_index_path,
                                                                        adjacency_matrix_path=adjacency_matrix_path,
                                                                        mirna_to_gene_matrix_path=mirna_to_gene_matrix_path,
                                                                        gene_gene=args.gene_gene,
                                                                        mirna_gene=args.mirna_gene,
                                                                        mirna_mirna=args.mirna_mirna,
                                                                        number_gene=args.num_gene,
                                                                        singleton=False)


print('Total # of samples:', train_data_all.shape[0])
print('Total # of features per sample:', train_data_all.shape[1])
print('Total # of classes:', len(np.unique(labels)))
print('Shape of adjacency matrix:', adj.shape)
print(labels)
#export adj to result folder for visualization
sp.save_npz(result_path + 'adj_matrix_used.npz', adj)
## filter data if a specific cancer type is selected
if args.cancer_subtype:
    train_data_all, labels = filter_data_by_cancer_type(cancer_subtype_label_path,
                                                        train_data_all,
                                                        expr_all_data)
print('After filtering, total # of samples:', train_data_all.shape[0])
print('After filtering, total # of classes:', len(np.unique(labels)))
print(labels)
## process the labels to make sure it starts from 0
from sklearn import preprocessing
from torch_sparse import SparseTensor

le = preprocessing.LabelEncoder()
labels = le.fit_transform(labels)  
# if not args.singleton:      
#     adj, train_data_all = removeZeroAdj(adj, train_data_all)

## adds the self connecting edges and convert the adj to edge_index format
adj_for_loss = adj.todense()
adj = adj/np.max(adj)
adj = adj.astype('float32')
adj.setdiag(0)
adj = adj + sp.eye(adj.shape[0])

adj = sp.coo_matrix(adj)
edge_index = torch.stack([torch.tensor(adj.row), torch.tensor(adj.col)], dim=0)
edge_weight = torch.Tensor(adj.data)
print('Total # of edges:', edge_index.shape[1])

## convert uniformly edge weight into multi-dimension edge attributes if needed.
if args.edge_attribute:
    edge_attribute = disassemble_edge_weights(edge_weight, edge_index, args.num_gene, args.num_omic)

# print(edge_weight.shape)



## split the training/val/test data using precomputed indices
if args.cancer_subtype is False:
    print('Using precomputed train/val/test splits with seed 42')
    print(args.split_seed)
    
    train_idx_path = path + "splits/splits_seed_" + str(args.split_seed) + "/common_trimmed_shuffle_index_train.tsv"
    val_idx_path   = path + "splits/splits_seed_" + str(args.split_seed) + "/common_trimmed_shuffle_index_val.tsv"
    test_idx_path  = path + "splits/splits_seed_" + str(args.split_seed) + "/common_trimmed_shuffle_index_test.tsv"

    train_idx = pd.read_csv(train_idx_path, sep="\t", header=None).iloc[:, 0].astype(np.int32).to_numpy()
    val_idx   = pd.read_csv(val_idx_path,   sep="\t", header=None).iloc[:, 0].astype(np.int32).to_numpy()
    test_idx  = pd.read_csv(test_idx_path,  sep="\t", header=None).iloc[:, 0].astype(np.int32).to_numpy()

    train_size = len(train_idx)  # serve dopo per decay_steps
    val_size   = len(val_idx)
    test_size  = len(test_idx)

    train_data = np.asarray(train_data_all, dtype=np.float32)[train_idx]
    val_data   = np.asarray(train_data_all, dtype=np.float32)[val_idx]
    test_data  = np.asarray(train_data_all, dtype=np.float32)[test_idx]

    train_labels = labels[train_idx]
    val_labels   = labels[val_idx]
    test_labels  = labels[test_idx]
else:
    ## split the training and test data
    shuffle_index = shuffle_index.astype(np.int32).reshape(-1)
    train_size, val_size = int(len(shuffle_index)* args.train_ratio), int(len(shuffle_index)* (1- args.test_ratio))
    train_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[0:train_size]]
    val_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[train_size:val_size]]
    test_data = np.asarray(train_data_all).astype(np.float32)[shuffle_index[val_size:]]
    train_labels = labels[np.array(shuffle_index[0:train_size])]
    val_labels = labels[np.array(shuffle_index[train_size:val_size])]
    test_labels = labels[np.array(shuffle_index[val_size:])]

print('Training samples:', train_data.shape[0])
print('Validation samples:', val_data.shape[0])
print('Testing samples:', test_data.shape[0])

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

dset_train = TensorDataset(train_data, train_labels)
train_loader = DataLoader(dset_train, batch_size = args.batch_size, shuffle = True)
dset_test = TensorDataset(test_data, test_labels)
test_loader = DataLoader(dset_test, shuffle = False)
dset_val = TensorDataset(val_data, val_labels)
val_loader = DataLoader(dset_val, batch_size = args.batch_size, shuffle = False)

print('train size:', len(dset_train))
print('val size:', len(dset_val))
print('test size:', len(dset_test))


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

print('Device', device)

if args.model == 'gcn':
    model = GCN(args.model, 
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 20% every 12 epochs"""
    if args.big_lr:
        lr = args.lr * pow(decay, float(global_step// decay_steps))
    else:
        lr = args.lr * (0.8 ** (epoch // 12))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
# optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr= args.lr)

l2_regularization = 5e-4

t_total_train = time.time()
# print the model structure
print(model)
# lists to track metrics per epoch
train_loss_history = []
train_acc_history = []
train_f1_macro_history = []
train_f1_weighted_history = []
val_loss_history = []
val_acc_history = []
val_f1_macro_history = []
val_f1_weighted_history = []

for epoch in range(args.epochs):
    ## update learning rate
    cur_lr = adjust_learning_rate(optimizer,epoch)

    ## start the timer and the criterion
    t_start = time.time()
    model.train()
    loss_all = 0.0
    accuracy_all = 0.0
    count = 0
    train_loss_sum = 0.0
    train_true = []
    train_pred = []

    print(f"train_loader: {len(train_loader)} batches per epoch")
    for i, (batch_x, batch_y) in enumerate(train_loader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        # print(batch_x.shape)
        # print(batch_y.shape)


        ## as gat or gatv2 not support static graph
        ## create the repeated edge_index for the batch
        count += 1
        batch_edge_index = edge_index.type(torch.int64)

        ## create the edge weight for the batch
        if args.edge_attribute:
            # batch_edge_weight = edge_attribute.type(torch.double)
            batch_edge_weight = edge_attribute
        else:
            # batch_edge_weight = edge_weight.type(torch.double)
            batch_edge_weight = edge_weight

        # print(batch_edge_weight.shape)
        ## concate batch numbers set of graph together
        for i in range(batch_y.shape[0] - 1):
            #print(i)
            if args.edge_weight and args.edge_attribute == False:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_weight], axis=0)
            elif args.edge_attribute:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_attribute], axis=0)
            batch_edge_index = torch.cat([batch_edge_index, edge_index+i*(args.num_gene+num_mirna)], axis=1)
            #print(batch_edge_index.size)

        # print(batch_edge_index.shape)
        # print(batch_edge_weight.shape)
        # print(batch_x.shape)
        # print(torch.max(batch_edge_index), torch.min(batch_edge_index))
        # print(torch.max(batch_edge_weight), torch.min(batch_edge_weight))

        # if args.edge_attribute == False:
        #     batch_edge_weight = batch_edge_weight.view(-1,1)
        # print(batch_edge_index.shape)
        # print(batch_edge_weight.shape)
        batch_edge_index = batch_edge_index.to(device)
        batch_edge_weight = batch_edge_weight.to(device)

        optimizer.zero_grad()
        if args.decoder:
            # print(args.edge_weight)
            x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
        else:
            out = model(batch_x, batch_edge_index, batch_edge_weight)
        # loss_batch = nn.CrossEntropyLoss()(out, batch_y)

        ## use different loss function with or without encoder
        if args.decoder:
            loss_batch = model.loss(x_reconstruct, batch_x, out, batch_y, l2_regularization)
        else:
            loss_batch = model.loss(batch_x.view(batch_x.size()[0], -1), batch_x, out, batch_y, l2_regularization)
        # cambiamento per accuracy 28/01
        pred = out.argmax(dim=1)
        bs = batch_y.size(0)

        train_loss_sum += loss_batch.item() * bs
        train_true.append(batch_y.detach().cpu())
        train_pred.append(pred.detach().cpu())

        #accuracy_batch = accuracy(out, batch_y)
        loss_batch.backward()
        if i == 0:
            logger.info(f"Batch 0")

        # --- INIZIO DEBUG GRADIENTI ---
        # Stampiamo solo all'inizio di ogni epoca per non intasare la console
        # put the values in a log file
        if i == 0 and epoch % 5 == 0:
            logger.info(f"--- Epoch {epoch+1} Start ---")
            total_norm = 0.0
            # print layer list
            for name, param in model.named_parameters():
                logger.info(f"Layer: {name}")
            logger.info(f"--- Gradient Information ---")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Calcola la norma del gradiente
                    param_norm = param.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
                    
                    # Calcola max e media
                    grad_mean = param.grad.data.mean().item()
                    grad_max = param.grad.data.max().item()
                    # Stampa layer interessanti (GNN vs Classifier)
                    logger.info(f"Layer: {name:<30} | Norm: {param_norm:.6f} | Mean: {grad_mean:.8f} | Max: {grad_max:.8f}")

            total_norm = total_norm ** 0.5
            logger.info(f"Total Gradient Norm: {total_norm:.6f}")            
            if total_norm < 1e-5:
                logger.warning("!!! ALLARME: VANISHING GRADIENT (Il modello non impara) !!!")
        # --- FINE DEBUG GRADIENTI ---
            print(f"Epoch {epoch}: Gradienti salvati in 'gradient_debug.log'")

        optimizer.step()
        # loss_all += batch_y.shape[0] * loss.item()
        loss_all += loss_batch.item()
        #accuracy_all += accuracy_batch
        global_step += bs

    # --- TRAIN metrics (sklearn) ---
    y_true_tr = torch.cat(train_true).numpy()
    y_pred_tr = torch.cat(train_pred).numpy()
    train_loss = train_loss_sum / len(train_loader.dataset)
    train_acc = accuracy_score(y_true_tr, y_pred_tr)
    train_f1_macro = f1_score(y_true_tr, y_pred_tr, average="macro", zero_division=0)
    train_f1_weighted = f1_score(y_true_tr, y_pred_tr, average="weighted", zero_division=0)
        
    ## test on the validation data
    model.eval()
    val_loss_sum = 0.0
    val_true = []
    val_pred = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            ## as gat or gatv2 not support static graph
            ## create the repeated edge_index for the batch
            batch_edge_index = edge_index.type(torch.int64)

            if args.edge_attribute:
                batch_edge_weight = edge_attribute
            else:
                batch_edge_weight = edge_weight

            for i in range(batch_y.shape[0] - 1):
                if args.edge_weight and args.edge_attribute == False:
                    batch_edge_weight = torch.cat([batch_edge_weight, edge_weight], axis=0)
                elif args.edge_attribute:
                    batch_edge_weight = torch.cat([batch_edge_weight, edge_attribute], axis=0)
                batch_edge_index = torch.cat([batch_edge_index, edge_index + i*(args.num_gene+num_mirna)], axis=1)

            batch_edge_index = batch_edge_index.to(device)
            batch_edge_weight = batch_edge_weight.to(device)

            if args.decoder:
                x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
                loss_val = model.loss(x_reconstruct, batch_x, out, batch_y, l2_regularization)
            else:
                out = model(batch_x, batch_edge_index, batch_edge_weight)
                loss_val = model.loss(batch_x.view(batch_x.size()[0], -1), batch_x, out, batch_y, l2_regularization)
            pred = out.argmax(dim=1)
            bs = batch_y.size(0)
            val_loss_sum += loss_val.item() * bs
            val_true.append(batch_y.detach().cpu())
            val_pred.append(pred.detach().cpu())

    # --- VAL metrics (sklearn) ---
    y_true_va = torch.cat(val_true).numpy()
    y_pred_va = torch.cat(val_pred).numpy()
    val_loss = val_loss_sum / len(val_loader.dataset)
    val_acc = accuracy_score(y_true_va, y_pred_va)
    val_f1_macro = f1_score(y_true_va, y_pred_va, average="macro", zero_division=0)
    val_f1_weighted = f1_score(y_true_va, y_pred_va, average="weighted", zero_division=0)
    print("Logging to wandb...")
    run.log({
        "Train Loss": train_loss,
        "Train Accuracy": train_acc,
        "Train F1 Macro": train_f1_macro,
        "Train F1 Weighted": train_f1_weighted,
        "Validation Loss": val_loss,
        "Validation Accuracy": val_acc,
        "Validation F1 Macro": val_f1_macro,
        "Validation F1 Weighted": val_f1_weighted,
        "Learning Rate": cur_lr,
        "Epoch": epoch + 1
    })
    # finish wandb logging for the epoch

    print(f'Epoch {epoch+1}/{args.epochs}:')
    print('lr:', cur_lr)
    t_stop = time.time() - t_start
    print('train:')
    print(f'epoch: {epoch}, avg_loss: {train_loss}, accuracy:{train_acc}, f1_macro:{train_f1_macro}, f1_weighted:{train_f1_weighted}')
    print('validation:')
    print(f'epoch: {epoch}, avg_loss: {val_loss}, accuracy:{val_acc}, f1_macro:{val_f1_macro}, f1_weighted:{val_f1_weighted}')
    print('training_time:',t_stop)
    # append metrics for plotting
    train_loss_history.append(float(train_loss))
    train_acc_history.append(float(train_acc))
    val_loss_history.append(float(val_loss))
    val_acc_history.append(float(val_acc))
    train_f1_macro_history.append(float(train_f1_macro))
    train_f1_weighted_history.append(float(train_f1_weighted))
    val_f1_macro_history.append(float(val_f1_macro))
    val_f1_weighted_history.append(float(val_f1_weighted))

    # if last epoch, save plots and metrics
    if epoch == args.epochs - 1:
        os.makedirs('result', exist_ok=True)
        # plot loss and accuracy
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(train_loss_history, label='train_loss')
        plt.plot(val_loss_history, label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(train_acc_history, label='train_acc')
        plt.plot(val_acc_history, label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join('result', 'training_validation_curves.png'), dpi=200)
        plt.close()

        # save metrics csv
        df_metrics = pd.DataFrame({
            'epoch': list(range(1, len(train_loss_history)+1)),
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'train_acc': train_acc_history,
            'val_acc': val_acc_history,
            'train_f1_macro': train_f1_macro_history,
            'val_f1_macro': val_f1_macro_history,
            'train_f1_weighted': train_f1_weighted_history,
            'val_f1_weighted': val_f1_weighted_history
        })

        df_metrics.to_csv(os.path.join('result', 'training_validation_metrics.csv'), index=False)
        print('Saved training/validation plots and metrics to result/')

        # plot acc and f1 curves separately
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(train_acc_history, label='train_acc')
        plt.plot(val_acc_history, label='val_acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(train_f1_macro_history, label='train_f1_macro')
        plt.plot(val_f1_macro_history, label='val_f1_macro')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Macro')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join('result', 'accuracy_f1_curves.png'), dpi=200)
        plt.close()
        print('Saved accuracy/F1 plots to result/')

run.finish()

def test(loader, num_classes):
    model.eval()

    test_true = []
    test_pred = []

    print(len(loader.dataset))
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        ## as gat or gatv2 not support static graph
        ## create the repeated edge_index for the batch
        batch_edge_index = edge_index.type(torch.int64)
        
        if args.edge_attribute:
            batch_edge_weight = edge_attribute
        else:
            batch_edge_weight = edge_weight

        for i in range(batch_y.shape[0] - 1):
            if args.edge_weight and args.edge_attribute == False:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_weight], axis=0)
            elif args.edge_attribute:
                batch_edge_weight = torch.cat([batch_edge_weight, edge_attribute], axis=0)
            batch_edge_index = torch.cat([batch_edge_index, edge_index+i*(args.num_gene+num_mirna)], axis=1)
        # print(tmp.size)

    # print(batch_edge_index.shape)

        batch_edge_index = batch_edge_index.to(device)
        batch_edge_weight = batch_edge_weight.to(device)

        if args.decoder:
            x_reconstruct, out = model(batch_x, batch_edge_index, batch_edge_weight)
        else:
            out = model(batch_x, batch_edge_index, batch_edge_weight)
        
        pred = out.argmax(dim=1)
        test_true.append(batch_y.detach().cpu())
        test_pred.append(pred.detach().cpu())

    
    ## print the final classification report 
    #classification_report = sklearn.metrics.classification_report(test_labels, np.argmax(np.asarray(predictions), 1), labels=range(num_classes))
    #print(classification_report)
    # export classification report to a text file
    #with open(os.path.join(result_path, 'classification_report.txt'), 'w') as f:
    #    f.write(classification_report)
    return test_true, test_pred

test_true, test_pred = test(test_loader, nclass)
# --- TEST metrics (sklearn) ---
y_true_te = torch.cat(test_true).numpy()
y_pred_te = torch.cat(test_pred).numpy()
test_acc = accuracy_score(y_true_te, y_pred_te)
test_f1_macro = f1_score(y_true_te, y_pred_te, average="macro", zero_division=0)
test_f1_weighted = f1_score(y_true_te, y_pred_te, average="weighted", zero_division=0)
print('Test set results:')
print(f'Accuracy: {test_acc}, F1 Macro: {test_f1_macro}, F1 Weighted: {test_f1_weighted}')

# Generate classification report
labels_range = list(range(nclass))
test_f1_micro = f1_score(y_true_te, y_pred_te, average="micro", zero_division=0)

# Get classification report as dict and string
report_dict = classification_report(y_true_te, y_pred_te, labels=labels_range, 
                                   output_dict=True, zero_division=0)
report_str = classification_report(y_true_te, y_pred_te, labels=labels_range, 
                                  zero_division=0)

# Calculate per-class metrics
cm_for_metrics = confusion_matrix(y_true_te, y_pred_te, labels=labels_range)
row_sums = cm_for_metrics.sum(axis=1).astype(float)
row_sums[row_sums == 0] = 1.0  # avoid division by zero

per_class_acc = np.diag(cm_for_metrics) / row_sums

# Save comprehensive test metrics to text file
test_metrics_path = os.path.join(result_path, 'test_metrics.txt')
with open(test_metrics_path, 'w') as f:
    f.write("=== OVERALL METRICS ===\n")
    f.write(f"       accuracy: {test_acc:.4f}\n")
    f.write(f"     f1 (micro): {test_f1_micro:.4f}\n")
    f.write(f"       f1_macro: {test_f1_macro:.4f}\n")
    f.write(f"    f1_weighted: {test_f1_weighted:.4f}\n")
    f.write("\n=== PER-CLASS METRICS ===\n")
    f.write(" class  precision     recall   accuracy\n")
    f.write("-" * 40 + "\n")
    
    for i in range(nclass):
        key = str(i)
        prec = report_dict[key]['precision']
        rec = report_dict[key]['recall']
        acc = per_class_acc[i]
        f.write(f"{i:6d}     {prec:.4f}     {rec:.4f}     {acc:.4f}\n")
    
    f.write("\n=== CLASSIFICATION REPORT (sklearn) ===\n")
    f.write(report_str)

print(f"Saved comprehensive test metrics to {test_metrics_path}")

# Save per-class metrics to CSV
per_class_df = pd.DataFrame({
    'class': list(range(nclass)),
    'precision': [report_dict[str(i)]['precision'] for i in range(nclass)],
    'recall': [report_dict[str(i)]['recall'] for i in range(nclass)],
    'f1-score': [report_dict[str(i)]['f1-score'] for i in range(nclass)],
    'support': [report_dict[str(i)]['support'] for i in range(nclass)],
    'accuracy': per_class_acc
})
per_class_csv_path = os.path.join(result_path, 'test_per_class_metrics.csv')
per_class_df.to_csv(per_class_csv_path, index=False)
print(f"Saved per-class metrics CSV to {per_class_csv_path}")
if args.cancer_subtype:
    if args.specific_type.lower() == 'brca':
        class_names = [
            "Basal", "Her2", "LumA", "LumB"
        ]
else:
    class_names = [
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
        "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19", "C20",
        "C21", "C22", "C23", "C25", "C26", "C27", "C28"
    ]

assert len(class_names) == nclass, f"len(class_names)={len(class_names)} != nclass={nclass}"

# --- Confusion matrix (counts + normalized) ---
labels = list(range(nclass))
cm = confusion_matrix(y_true_te, y_pred_te, labels=labels)

# DataFrame con nomi (righe=true, colonne=pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

cm_counts_path = os.path.join(result_path, "confusion_matrix_counts.csv")
cm_df.to_csv(cm_counts_path, index=True)
print(f"Saved confusion matrix (counts) to {cm_counts_path}")
labels = list(range(nclass))
cm = confusion_matrix(y_true_te, y_pred_te, labels=labels)

# --- salva CSV con etichette (righe=true, colonne=pred) ---
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
cm_counts_path = os.path.join(result_path, "confusion_matrix_counts.csv")
cm_df.to_csv(cm_counts_path, index=True)

# --- plot PNG (counts) ---
plt.figure(figsize=(12, 10))
im = plt.imshow(cm, interpolation="nearest", aspect="auto")
plt.title("Confusion Matrix (Counts)")
plt.colorbar(im)

tick_font = 8
plt.xticks(np.arange(nclass), class_names, rotation=90, fontsize=tick_font)
plt.yticks(np.arange(nclass), class_names, fontsize=tick_font)
plt.xlabel("Predicted")
plt.ylabel("True")

plt.xlim(-0.5, nclass - 0.5)
plt.ylim(nclass - 0.5, -0.5)

plt.tight_layout()
cm_counts_png = os.path.join(result_path, "confusion_matrix_counts.png")
plt.savefig(cm_counts_png, dpi=300)
plt.close()

print("Saved confusion matrix to:")
print(" -", cm_counts_path)
print(" -", cm_counts_png)