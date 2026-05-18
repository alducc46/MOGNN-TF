from tokenize import Double
import numpy as np
import pandas as pd
import scipy.sparse as sp
from random import sample
import torch
import argparse
import sklearn.metrics
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

## old data loading functions from the gnn model
def high_variance_expression_gene(expression_variance_path, non_null_path, num_gene, singleton=False):
    ## filter gene list by expression variance
    gene_variance = pd.read_csv(expression_variance_path, sep='\t', index_col=0, header=0)

    if singleton:
        non_null_row = pd.read_csv(non_null_path, sep=',', header=0)
        gene_variance['id'] = range(gene_variance.shape[0])
#         print(non_null_row['gene'])
        gene_variance_non_null = gene_variance.loc[gene_variance.index.isin(non_null_row['gene']),:]
        gene_list = gene_variance_non_null.nlargest(num_gene, 'variance').index
#         print(gene_list)
        gene_variance_non_null.index = gene_variance_non_null['id']
        gene_list_index = gene_variance_non_null.nlargest(num_gene, 'variance').index
    else:
        ## load expression data
        gene_list = gene_variance.nlargest(num_gene, 'variance').index
        gene_variance.index = range(gene_variance.shape[0])
        gene_list_index = gene_variance.nlargest(num_gene, 'variance').index
    return gene_list, gene_list_index
# la matrice mirna mirna non è simmetrica
def get_mirna_inner_connection(mirna_connection):
    mirna_connection_row = []
    mirna_connection_col = []
    print("Building miRNA-miRNA connections...")
    print("miRNA-gene connection matrix shape:", mirna_connection.shape)
    for i in range(mirna_connection.shape[0]):
        current_indices = np.nonzero(mirna_connection[i,])
        # print(current_indices)
        column_indexes = current_indices[1]
        if len(column_indexes) > 1:
            for x in range(len(column_indexes)):
                for y in range(x, len(column_indexes)):
                    mirna_connection_row += [column_indexes[x]]
                    mirna_connection_col += [column_indexes[y]]
    
    mirna_connection_data = [1] * len(mirna_connection_row)
    mirna_connection_adj = sp.csr_matrix((mirna_connection_data, (mirna_connection_row, mirna_connection_col)),shape=(mirna_connection.shape[1], mirna_connection.shape[1]))
    mirna_index = mirna_connection_adj.nonzero()
    mirna_row = mirna_index[0]
    mirna_col = mirna_index[1]
    mirna_data = [1] * len(mirna_row)
    mirna_adj = sp.csr_matrix((mirna_data, (mirna_row, mirna_col)),shape=(mirna_connection.shape[1], mirna_connection.shape[1]))
    # check symmetry
    print("check symmetry miRNA-miRNA adjacency matrix before symmetrization:", (mirna_adj != mirna_adj.T).nnz)
    mirna_sym = mirna_adj + mirna_adj.T
    mirna_sym[mirna_sym > 1] = 1
    # check symmetry
    A = mirna_sym
    if not sp.isspmatrix(A):
        A = sp.csr_matrix(np.asarray(A))   # np.asarray evita numpy.matrix
    A = A.tocsr()
    print("check symmetry miRNA-miRNA adjacency matrix after symmetrization:", (A != A.T).nnz)
    mirna_adj = mirna_sym
    return(mirna_adj.toarray())


def load_exp_and_real_mirna_data(expression_data_path, mirna_data_path):
    ## load multi-omics data
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
    mirna_data = pd.read_csv(mirna_data_path, sep='\t', index_col=0, header=0)
    return expression_data, mirna_data

def load_exp_cnv_and_real_mirna_data(expression_data_path, cnv_data_path, mirna_data_path):
    ## load multi-omics data
    expression_data = pd.read_csv(expression_data_path, sep='\t', index_col=0, header=0)
    cnv_data = pd.read_csv(cnv_data_path, sep='\t', index_col=0, header=0)
    cnv_data = cnv_data.drop(['icluster_cluster_assignment','sample'], axis=1)
    cnv_data = (cnv_data - cnv_data.min().min()) / (cnv_data.max().max() - cnv_data.min().min())
    mirna_data = pd.read_csv(mirna_data_path, sep='\t', index_col=0, header=0)
    return expression_data, cnv_data, mirna_data

def down_sampling_exp_and_real_mirna_data(expression_variance_path, 
                                            expression_data, 
                                            mirna_data, 
                                            omic_mode, 
                                            non_null_index_path, 
                                             
                                            adjacency_matrix_path, 
                                            mirna_to_gene_matrix_path, 
                                            gene_gene, 
                                            mirna_gene, 
                                            mirna_mirna, 
                                            number_gene,  
                                            singleton=False):
    ## obtain high varaince gene list
    high_variance_gene_list, high_variance_gene_index = high_variance_expression_gene(expression_variance_path, non_null_index_path, number_gene, singleton)

    ## get labels before filtering columns
    labels = expression_data['icluster_cluster_assignment']
    labels = labels - 1
    
    if expression_data.shape[0] == mirna_data.shape[0]:
        print('Exp and miRNA sample numbers match.')
    # print(mirna_influence_data.shape)
    ## filter multi-omics data by gene list
    expression_data = expression_data.loc[:,high_variance_gene_list]
    expression_data.index = range(expression_data.shape[0])
    
    ## involves mirna data in the following
    if omic_mode > 0:
        mirna_data.index = range(mirna_data.shape[0])
        data = pd.concat([expression_data, mirna_data], axis=1)
        num_samples = mirna_data.shape[0]
    ## concatenate expr and mirna
        data = np.asarray(data).reshape(num_samples, -1 ,1)
        print(data.shape)
    else:
        data = np.asarray(expression_data).reshape(expression_data.shape[0], -1 ,1)

    ## load adjacency matrix
    if gene_gene:
        gene_gene_adj = sp.load_npz(adjacency_matrix_path)
        print('Gene Adjacency matrix max value:', np.max(gene_gene_adj.data))
        print('Gene Adjacency matrix min value:', np.min(gene_gene_adj.data))
        gene_gene_adj_mat = gene_gene_adj.todense()
        gene_gene_adj_mat = gene_gene_adj_mat / (gene_gene_adj_mat.max() - gene_gene_adj_mat.min())
        gene_gene_adj_selected = gene_gene_adj_mat[high_variance_gene_index,:]
        gene_gene_adj_selected = gene_gene_adj_selected[:,high_variance_gene_index]
    else:
        gene_gene_adj_selected = np.identity(number_gene)
    
    ## load mirna_to_gene matrix
    if mirna_gene or mirna_mirna:
        mirna_gene_adj = sp.load_npz(mirna_to_gene_matrix_path)
        print('miRNA-Gene matrix max value:', np.max(mirna_gene_adj.data))
        print('miRNA-Gene matrix min value:', np.min(mirna_gene_adj.data))
        mirna_gene_adj = mirna_gene_adj.todense()
        mirna_gene_adj_selected = mirna_gene_adj[high_variance_gene_index,:]
    else:
        mirna_gene_adj_selected = np.zeros((number_gene,100))

    if omic_mode == 2:
        top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
        if mirna_mirna:
            print(mirna_gene_adj_selected.shape)
            print(get_mirna_inner_connection(mirna_gene_adj_selected).shape)
            bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), get_mirna_inner_connection(mirna_gene_adj_selected)), axis=1)
        else:
            bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), np.identity(100)), axis=1)
    
        supra_adj = np.concatenate((top_supra_adj, bottom_supra_adj), axis=0)

        ## convert the dense matrix back to sparse matrix
        supra_adj = sp.csr_matrix(supra_adj)

        if singleton:
            print('including singleton')
            supra_adj = supra_adj + sp.eye(supra_adj.shape[0])
    elif omic_mode == 0:
        supra_adj = sp.csr_matrix(gene_gene_adj_selected)
    elif omic_mode == 1:
        supra_adj = sp.csr_matrix(get_mirna_inner_connection(mirna_gene_adj_selected))

    #shuffle_index = pd.read_csv(shuffle_index_path, sep='\t', index_col=0, header=0)
    
    return supra_adj, np.asarray(data), labels.to_numpy()

def down_sampling_exp_cnv_and_real_mirna_data(expression_variance_path, 
                                            expression_data, 
                                            cnv_data,
                                            mirna_data, 
                                            omic_mode, 
                                            non_null_index_path, 
                                             
                                            adjacency_matrix_path, 
                                            mirna_to_gene_matrix_path, 
                                            gene_gene, 
                                            mirna_gene, 
                                            mirna_mirna, 
                                            number_gene,  
                                            singleton=False):
    ## obtain high varaince gene list
    high_variance_gene_list, high_variance_gene_index = high_variance_expression_gene(expression_variance_path, non_null_index_path, number_gene, singleton)

    ## get labels before filtering columns
    labels = expression_data['icluster_cluster_assignment']
    labels = labels - 1
    
    if expression_data.shape[0] == mirna_data.shape[0]:
        print('Exp and miRNA sample numbers match.')
    ## filter multi-omics data by gene list
    expression_data = expression_data.loc[:,high_variance_gene_list]
    cnv_data = cnv_data.loc[:,high_variance_gene_list]
    expression_data.index = range(expression_data.shape[0])
    mirna_data.index = range(mirna_data.shape[0])
    cnv_data.index = range(cnv_data.shape[0])
    print("cnv_data.shape:", cnv_data.shape)
    print("expression_data.shape:", expression_data.shape)
    print("mirna_data.shape:", mirna_data.shape)

    ##  only pad CNV data when using Exp, CNV and miRNA
    if omic_mode == 4:
        cnv_padding = pd.DataFrame(np.zeros((mirna_data.shape[0],mirna_data.shape[1])))
        cnv_data_padded = pd.concat([cnv_data, cnv_padding], axis=1)

        data = pd.concat([expression_data, mirna_data], axis=1)
        
        num_samples = mirna_data.shape[0]
        ## concatenate expr and mirna
        data= np.asarray(data).reshape(num_samples, -1 ,1)
        cnv_data_padded = np.asarray(cnv_data_padded).reshape(num_samples, -1 ,1)
        data = np.concatenate([data,cnv_data_padded], axis=2)
        print(data.shape)
    else:
        num_samples = expression_data.shape[0]
        data = np.array(expression_data).reshape(num_samples, -1 ,1)
        print(data.shape)
        cnv_data = np.asarray(cnv_data).reshape(num_samples, -1 ,1)
        print(data.shape)
        data = np.concatenate([data,cnv_data], axis=2)
        print(data.shape)

    ## load adjacency matrix
    if gene_gene:
        print("Utils_li.py - Loading gene-gene adjacency matrix...")
        gene_gene_adj = sp.load_npz(adjacency_matrix_path)
        A = gene_gene_adj
        if not sp.isspmatrix(A):
            A = sp.csr_matrix(np.asarray(A))   # np.asarray evita numpy.matrix
        A = A.tocsr()
        print("check symmetry gene-gene adjacency matrix after normalization:", (A != A.T).nnz)
        print('Original Global Max:', np.max(gene_gene_adj.data))
        print('Original Global Min:', np.min(gene_gene_adj.data))
        print("Gene adj matrix shape:", gene_gene_adj.shape)
        # Che normalizzazione sta usando??
        gene_gene_adj_mat = gene_gene_adj.todense()
        #gene_gene_adj_mat = gene_gene_adj_mat / (gene_gene_adj_mat.max() - gene_gene_adj_mat.min())
        gene_gene_adj_selected = gene_gene_adj_mat[high_variance_gene_index,:]
        gene_gene_adj_selected = gene_gene_adj_selected[:,high_variance_gene_index]
        #min max normalization
        print("Gene adj matrix max value before normalization:", np.max(gene_gene_adj_selected))
        print("Gene adj matrix min value before normalization:", np.min(gene_gene_adj_selected))
        gene_gene_adj_selected = (gene_gene_adj_selected - gene_gene_adj_selected.min()) / (gene_gene_adj_selected.max() - gene_gene_adj_selected.min())
        # check the simmerty of matrix after normalization
        A = gene_gene_adj_selected
        if not sp.isspmatrix(A):
            A = sp.csr_matrix(np.asarray(A))   # np.asarray evita numpy.matrix
        A = A.tocsr()
        print("check symmetry gene-gene adjacency matrix after normalization:", (A != A.T).nnz)
        #export the matric representation in a file for debug
        np.savetxt("gene_gene_adj_selected_normalized.csv", gene_gene_adj_selected, delimiter=",")
        

        print("Gene adj matrix max value after normalization:", np.max(gene_gene_adj_selected))
        print("Gene adj matrix min value after normalization:", np.min(gene_gene_adj_selected))
        # scrivi il conteggio per ogni valore nella matrice
        print("Non-zero count:", np.count_nonzero(gene_gene_adj_selected))
        print("Total elements:", gene_gene_adj_selected.size)
        # --- BLOCCO DI STAMPA ---
        # Estrae solo i valori > 0
        non_zero_values = gene_gene_adj_selected[gene_gene_adj_selected > 0]
        
        print("tipo di non_zero_values:", type(non_zero_values))
        # stampa il numero di valori non-zero trovati, tipo nupy.dmatrix
        print("Numero di valori non-zero trovati:", non_zero_values.shape[0])

        
        # Stampa tutti i valori (ordinati per capire meglio la distribuzione)
        print("Lista dei valori non-zero (ordinati):")
        print(np.sort(non_zero_values))
    else:
        gene_gene_adj_selected = np.identity(number_gene)
    
    ## load mirna_to_gene matrix
    if mirna_gene or mirna_mirna:
        mirna_gene_adj = sp.load_npz(mirna_to_gene_matrix_path)
        print("mirna_to_gene_matrix shape:", mirna_gene_adj.shape)
        print('miRNA-Gene matrix max value:', np.max(mirna_gene_adj.data))
        print('miRNA-Gene matrix min value:', np.min(mirna_gene_adj.data))
        mirna_gene_adj = mirna_gene_adj.todense()
        mirna_gene_adj_selected = mirna_gene_adj[high_variance_gene_index,:]
        print("mirna_gene_adj_selected shape:", mirna_gene_adj_selected.shape)
        print("mirna_gene_adj_selected max value:", np.max(mirna_gene_adj_selected))
        print("mirna_gene_adj_selected min value:", np.min(mirna_gene_adj_selected))
        print("Non-zero count in mirna_gene_adj_selected:", np.count_nonzero(mirna_gene_adj_selected))
        print("Total elements in mirna_gene_adj_selected:", mirna_gene_adj_selected.size)

    else:
        mirna_gene_adj_selected = np.zeros((number_gene,100))

    if omic_mode > 3:
        if mirna_mirna:
            print("Mirna-mirna connections enabled.")
            print(mirna_gene_adj_selected.shape)
            A = get_mirna_inner_connection(mirna_gene_adj_selected)

            print("check symmetry miRNA-miRNA adjacency matrix after construction:", np.any(A != A.T))
            print("miRNA-miRNA adj matrix shape:", A.shape)
            A = np.maximum(A, A.T)
            print("check symmetry miRNA-miRNA adjacency matrix after symmetrization:", np.any(A != A.T))

            if mirna_gene:
                top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
                bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), get_mirna_inner_connection(mirna_gene_adj_selected)), axis=1)
            else:
                top_supra_adj = np.concatenate((gene_gene_adj_selected, np.zeros((number_gene,100))), axis=1)
                bottom_supra_adj = np.concatenate((np.transpose(np.zeros((number_gene,100))), get_mirna_inner_connection(mirna_gene_adj_selected)), axis=1)
        else:
            print("Mirna-mirna connections disabled.")
            if mirna_gene:
                top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
                bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), np.identity(100)), axis=1)
            else:
                top_supra_adj = np.concatenate((gene_gene_adj_selected, mirna_gene_adj_selected), axis=1)
                bottom_supra_adj = np.concatenate((np.transpose(mirna_gene_adj_selected), np.identity(100)), axis=1)

        supra_adj = np.concatenate((top_supra_adj, bottom_supra_adj), axis=0)

        ## convert the dense matrix back to sparse matrix
        supra_adj = sp.csr_matrix(supra_adj)
    elif omic_mode == 3:
        supra_adj = sp.csr_matrix(gene_gene_adj_selected)

    if singleton:
        print('including singleton')
        supra_adj = supra_adj + sp.eye(supra_adj.shape[0])

    # del features['iCluster']
    #shuffle_index = pd.read_csv(shuffle_index_path, sep='\t', index_col=0, header=0)
    # print(shuffle_index.shape)
    print("Data shape after down-sampling:", data.shape)
    # check symmerty supra_adj
    A = supra_adj
    if not sp.isspmatrix(A):
        A = sp.csr_matrix(np.asarray(A))   # np.asarray evita numpy.matrix
    A = A.tocsr()
    print("check symmetry supra adjacency matrix after construction:", (A != A.T).nnz)
    print("Supra adj matrix max value after construction:", np.max(supra_adj))
    print("Supra adj matrix min value after construction:", np.min(supra_adj))
    print("End of Utils_li")
    return supra_adj, np.asarray(data), labels.to_numpy()

def dropout_data(data, labels, drop_out=0.6):
    dropout_index = sample(range(len(labels)), round(len(labels)*drop_out))
    dropped_data = data[dropout_index,:]
    dropped_labels = labels[dropout_index]
    return dropped_data, dropped_labels

def accuracy(output, labels): # average of each batch 
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def calculate_f1(preds, labels, average='weighted'):
    """
    Calcola F1 Score gestendo automaticamente Tensor (CPU/GPU) e Numpy.
    
    Args:
        preds: Tensor (logits [N, C] o classi [N]) oppure np.array
        labels: Tensor (classi [N]) oppure np.array (Ground Truth)
        average: 'weighted' (consigliato per classi sbilanciate), 'macro', 'micro'
    """
    
    # 1. Gestione Tensori PyTorch -> Numpy
    if isinstance(preds, torch.Tensor):
        # Se sono logits (es. shape [64, 28]), prendiamo l'argmax
        if preds.ndim > 1:
            preds = preds.argmax(dim=1)
        preds = preds.detach().cpu().numpy()
        
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
        
    # 2. Controllo coerenza forme
    # Assicuriamoci che siano array 1D
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()

    # 3. Calcolo effettivo
    return sklearn.metrics.f1_score(labels, preds, average=average, zero_division=0)

def edge_filter(adj, top=10, which_axis=1):
    adj = np.array(adj)
    print(adj.shape)
    new_adj = adj * (np.argsort(np.argsort(adj,axis=which_axis)) >= adj.shape[1] - top)
    return(new_adj)

def disassemble_edge_weights(edge_weights, edge_index, num_gene, num_attributes):
    edge_index_transposed = edge_index.T
    edge_attributes = np.zeros((edge_index.shape[1], num_attributes))
    for idx, x in enumerate(edge_index_transposed):
        # print(x)
        if x[0] < num_gene and x[1] < num_gene: ## gene-gene connection
            edge_attributes[idx,0] = edge_weights[idx]
        elif x[0] < num_gene and x[1] >= num_gene: ## mirna-gene connection
            edge_attributes[idx,1] = edge_weights[idx]
        elif x[0] >= num_gene and x[1] < num_gene: ## mirna-gene connection
            edge_attributes[idx,1] = edge_weights[idx]
        else: ## mirna-mirna connections
            edge_attributes[idx,0] = edge_weights[idx]
    return(torch.Tensor(edge_attributes))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def omic_mode_translation(omic_mode):
    if omic_mode == 0:
        print('Using Expression Data Only.')
        return 1
    elif omic_mode == 1:
        print('Using miRNA Data Only.')
        return 1
    elif omic_mode == 2:
        print('Using Expression and miRNA Data.')
        return 2
    elif omic_mode == 3:
        print('Using Expression and CNV Data.')
        return 2
    elif omic_mode == 4:
        print('Using Expression, CNV and miRNA Data.')
        return 3
    
def validate_network_choice(omic_mode, gene_gene, mirna_gene, mirna_mirna):
    if omic_mode == 0:
        if mirna_gene or mirna_mirna:
            print('miRNA-Gene or miRNA-miRNA network not available when only using expression data.')
            return True, False, False, 0
        else:
            return gene_gene, mirna_gene, mirna_mirna, 0
    elif omic_mode == 1:
        if gene_gene:
            print('Gene-Gene not available when only using miRNA data.')
            return False, True, True, 100
        else:
            return gene_gene, mirna_gene, mirna_mirna, 100
    elif omic_mode == 3:
        if mirna_gene or mirna_mirna:
            print('miRNA-Gene or miRNA-miRNA network not available when using expression and CNV data.')
            return True, False, False, 0
        else:
            return gene_gene, mirna_gene, mirna_mirna, 0
    else:
        return gene_gene, mirna_gene, mirna_mirna, 100

def filter_data_by_cancer_type(cancer_subtype_label_path, data, expression_data):
    print('Filtering data by cancer subtype labels from:', cancer_subtype_label_path)
    cancer_subtype_label = pd.read_csv(cancer_subtype_label_path, sep=',', header=0)
    cancer_subtype_label = cancer_subtype_label[cancer_subtype_label['patient'].isin(expression_data['sample'].tolist())]
    expression_index = expression_data[expression_data['sample'].isin(cancer_subtype_label['patient'].tolist())]['sample']

    ## check if the order of the patient is the same
    for i in range(cancer_subtype_label.shape[0]):
        if expression_index.iloc[i] != cancer_subtype_label.iloc[i,0]:
            print('Mismatch in patients!')
            quit()

    subtype_sample_index = expression_data['sample'].isin(cancer_subtype_label['patient'].tolist())

    data = data[subtype_sample_index]
    labels = cancer_subtype_label['icluster_cluster_assignment']

    return np.asarray(data), labels.to_numpy()

def get_all_metrics(true_labels, pred_labels, num_classes):
    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    # metriche globali
    acc = sklearn.metrics.accuracy_score(true_labels, pred_labels)
    f1_micro = f1_score(true_labels, pred_labels, average="micro")
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_weighted = f1_score(true_labels, pred_labels, average="weighted")

    global_metrics = {
        "accuracy": acc,
        "f1": f1_micro,          # F1 micro (sull'insieme)
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    # metriche per classe
    report = classification_report(
        true_labels,
        pred_labels,
        labels=range(num_classes),
        output_dict=True,
        zero_division=0,
    )
    
    confusion_mat = confusion_matrix(
        true_labels,
        pred_labels,
        labels=range(num_classes),
    )

    # metriche per classe: precision, recall + "accuratezza per classe"
    per_class_metrics = {}

    # accuratezza per classe = TP / (TP + FN) = elemento diagonale / somma riga
    row_sums = confusion_mat.sum(axis=1).astype(float)
    row_sums[row_sums == 0] = 1.0  # per sicurezza, evitare divisioni per zero

    for cls_idx in range(num_classes):
        key = str(cls_idx)
        cls_stats = report.get(key, {})

        precision = cls_stats.get("precision", 0.0)
        recall = cls_stats.get("recall", 0.0)

        # "accuracy" della singola classe (in pratica coincide con la recall di quella classe)
        class_accuracy = confusion_mat[cls_idx, cls_idx] / row_sums[cls_idx]

        per_class_metrics[cls_idx] = {
            "precision": precision,
            "recall": recall,
            "accuracy": class_accuracy,
        }

    return global_metrics, per_class_metrics, confusion_mat, report

import os
import pandas as pd
import matplotlib.pyplot as plt


def save_confusion_matrix_pretty(confusion_mat, filepath, class_names):
    """
    Salva una confusion matrix in stile 'paper':
      - normalizzata per riga (valori in [0,1]) annotati nelle celle
      - etichette delle classi prese da `class_names`
      - colonna finale con la somma (conteggi) per classe ("SUM")
      - CSV + PNG

    Parameters
    ----------
    confusion_mat : np.ndarray
        Matrice di confusione (n_class x n_class) con i conteggi.
    filepath : str
        Path del file di metrics txt: lo uso come base per cm.csv e cm.png.
    class_names : list[str]
        Lista dei nomi delle classi (stessa lunghezza di confusion_mat).
        Esempio: ["C1","C2","C3",...,"C23","C25","C26","C27","C28"].
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    base, _ = os.path.splitext(filepath)
    csv_path = base + "cm.csv"
    png_path = base + "cm.png"

    class_names = list(class_names)
    num_classes = len(class_names)

    # ------------------  CSV (con colonna SUM)  ------------------
    row_sums = confusion_mat.sum(axis=1)
    cm_df = pd.DataFrame(
        confusion_mat,
        index=[f"true_{c}" for c in class_names],
        columns=[f"pred_{c}" for c in class_names],
    )
    cm_df["SUM"] = row_sums
    cm_df.to_csv(csv_path)

    # ------------------  immagine tipo quella che hai mandato  ------------------
    confusion_mat = confusion_mat.astype(float)
    row_sums_safe = row_sums.copy().astype(float)
    row_sums_safe[row_sums_safe == 0] = 1.0
    cm_norm = confusion_mat / row_sums_safe[:, None]

    fig, ax = plt.subplots(figsize=(num_classes * 0.5, num_classes * 0.5))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Ground-truth")

    # spazio per la colonna SUM
    ax.set_xlim(-0.5, num_classes + 0.5)

    # valori nelle celle
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_norm[i, j]
            text = "0" if val == 0 else f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6)

    # colonna SUM a destra (conteggi grezzi)
    ax.text(num_classes, -0.8, "SUM", ha="center", va="center", fontsize=8)
    for i in range(num_classes):
        ax.text(num_classes, i, str(int(row_sums[i])), ha="center", va="center", fontsize=7)

    # griglia tipo tabella
    ax.set_xticks(np.arange(-0.5, num_classes + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, num_classes + 0.5, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    return csv_path, png_path


def save_metrics_pretty_txt(global_metrics,
                            per_class_metrics,
                            clf_report,
                            filepath,
                            num_classes,
                            class_names=None):
    """
    Scrive un file di testo leggibile con:
      - metriche globali
      - tabella per classe (precision, recall, accuracy)
      - matrice di confusione tabellare
    """
    import os
    import numpy as np

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    with open(filepath, "w") as f:
        # ---------- overall ----------
        f.write("=== OVERALL METRICS ===\n")
        f.write(f"{'accuracy':>15}: {global_metrics['accuracy']:.4f}\n")
        f.write(f"{'f1 (micro)':>15}: {global_metrics['f1']:.4f}\n")
        f.write(f"{'f1_macro':>15}: {global_metrics['f1_macro']:.4f}\n")
        f.write(f"{'f1_weighted':>15}: {global_metrics['f1_weighted']:.4f}\n\n")

        # ---------- per-class ----------
        f.write("=== PER-CLASS METRICS ===\n")
        f.write(f"{'class':>6} {'precision':>10} {'recall':>10} {'accuracy':>10}\n")
        f.write("-" * 40 + "\n")
        for cls_idx in range(num_classes):
            m = per_class_metrics.get(cls_idx, {})
            f.write(
                f"{class_names[cls_idx]:>6} "
                f"{m.get('precision', 0.0):10.4f} "
                f"{m.get('recall', 0.0):10.4f} "
                f"{m.get('accuracy', 0.0):10.4f}\n"
            )

        f.write("\n")
        # ---------- CLASSIFICATION REPORT ----------
        f.write("=== CLASSIFICATION REPORT (sklearn) ===\n")

        # Se è una stringa, la scrivo così com'è
        if isinstance(clf_report, str):
            f.write(clf_report)
        else:
            # Presumo sia un dict (output_dict=True)
            # Stampo almeno le righe per ciascuna classe
            f.write(f"{'class':>6} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}\n")
            f.write("-" * 60 + "\n")
            for cls_idx in range(num_classes):
                stats = clf_report.get(str(cls_idx), {})
                f.write(
                    f"{class_names[cls_idx]:>6} "
                    f"{stats.get('precision', 0.0):10.4f} "
                    f"{stats.get('recall', 0.0):10.4f} "
                    f"{stats.get('f1-score', 0.0):10.4f} "
                    f"{stats.get('support', 0):10.0f}\n"
                )

            # opzionale: macro avg / weighted avg se presenti
            for key in ["macro avg", "weighted avg"]:
                if key in clf_report:
                    stats = clf_report[key]
                    f.write(
                        f"{key:>12} "
                        f"{stats.get('precision', 0.0):10.4f} "
                        f"{stats.get('recall', 0.0):10.4f} "
                        f"{stats.get('f1-score', 0.0):10.4f} "
                        f"{stats.get('support', 0):10.0f}\n"
                    )


