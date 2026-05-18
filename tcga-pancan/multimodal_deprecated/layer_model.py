import numpy as np
import pandas as pd
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
# from torch.utils.loader import DataLoader

# from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv, GATv2Conv, ChebConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import MNISTSuperpixels

import torch_geometric.transforms as T

# num_genes = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## define the GAT class
class GAT(torch.nn.Module):
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(GAT, self).__init__()
        self.debug = False
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.head = 8
        #self.head = 4
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)

        if method == 'gatv2':
            if self.edge_attributes:
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head, edge_dim=2)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head, edge_dim=2)
            elif self.edge_weights:
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head, edge_dim=1)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head, edge_dim=1)
            else:
                self.conv1 = GATv2Conv(self.raised_dimension, self.hid, heads=self.head)
                self.conv2 = GATv2Conv(self.hid * self.head, self.hid, heads=self.head)

        elif method == 'gat':
            if self.edge_attributes:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head, edge_dim=2)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head, edge_dim=2)
            elif self.edge_weights:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head, edge_dim=1)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head, edge_dim=1)
            else:
                self.conv1 = GATConv(self.raised_dimension, self.hid, heads=self.head)
                self.conv2 = GATConv(self.hid * self.head, self.hid, heads=self.head)

        self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid * self.head
        #print(self.linear_input)

        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        if self.parallel:

            parallel_input = self.raised_dimension*(self.num_gene+self.num_mirna)

            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            self.classifier = nn.Linear(self.concate_layer*2, num_classes)
        else:
            self.classifier = nn.Linear(self.concate_layer, num_classes)

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            return x
        else:
            return x
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene+self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        batches = x.shape[0]
        num_node = x.shape[1]
        #print("[GAT] Input x shape:", x.shape)
        # Pre-convolution linear transforms
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]
            #print("[GAT] x_exp_mirna shape:", x_exp_mirna.shape)
            #print("[GAT] x_cnv shape:", x_cnv.shape)

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            #print("[GAT] x_cnv after slicing shape:", x_cnv.shape)
            x_exp = x_exp_mirna[:,:-100]
            #print("[GAT] x_exp after slicing shape:", x_exp.shape)

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            #print("[GAT] x_cnv after view shape:", x_cnv.shape)
            #print("[GAT] x_exp after view shape:", x_exp.shape)
            #print("exp gene0:", x_exp[0, 0, 0].item())
            #print("exp gene1:", x_exp[0, 1, 0].item())
            #print("cnv gene0:", x_cnv[0, 0, 0].item())
            x_gene = torch.cat([x_exp,x_cnv],dim=2)
            #print("[GAT] x_gene after concatenation shape:", x_gene.shape)

            #print("cat node0:",  x_gene[0, 0, 0].item())    # exp gene0
            #print("cat node1:",  x_gene[0, 1, 0].item())    # exp gene1
            #print("cat node700:",x_gene[0, 700, 0].item())    # cnv gene0
            # --- BLOCCO DEBUG DA INSERIRE ---
            #print("\n--- DEBUG VERIFICA ERRORE ---")
            #print(f"Shape attuale (errata): {x_gene.shape}") 
            # Se fosse corretto, x_gene[0, 0] dovrebbe avere 2 valori (Exp e CNV del gene 0).
            # Invece vedrai che ha solo 1 valore.
            #print(f"Valore nodo 0 (Gene 0 Exp): {x_gene[0, 0]}")
            #print(f"Valore nodo 700 (Gene 0 CNV finito in fondo): {x_gene[0, 700]}")
            
            # Simuliamo cosa succede quando fai .view(-1, 2) dopo
            # Questo mostrerà che il modello accoppierà il Gene 0 (Exp) con il Gene 1 (Exp)
            # invece di Gene 0 (Exp) con Gene 0 (CNV).
            #x_test_view = x_gene.view(-1, 2)
            ##print(f"Cosa vede il modello dopo il view (ERRORE): {x_test_view[0]}")
            ##print("-----------------------------\n")
            # --------------------------------

            x_gene = x_gene.view(-1,self.num_features)
            #print("[GAT] x_gene after view shape:", x_gene.shape)
            #print("view row0:", x_gene[0].tolist())

            x_mirna = x_exp_mirna[:,-100:]
            #print("[GAT] x_mirna before flatten shape:", x_mirna.shape)

            x_mirna = torch.flatten(x_mirna)
            #print("[GAT] x_mirna after flatten shape:", x_mirna.shape)
            x_mirna = x_mirna.view(-1, 1)
            #print("[GAT] x_mirna after view shape:", x_mirna.shape)

            x_gene = self.pre_conv_linear_gene(x_gene)
            #print("[GAT] x_gene after pre_conv_linear_gene shape:", x_gene.shape)
            x_gene = F.relu(x_gene)
            #print("[GAT] x_gene after ReLU shape:", x_gene.shape)

            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            #print("[GAT] x_mirna after pre_conv_linear_mirna shape:", x_mirna.shape)
            x_mirna = F.relu(x_mirna)
            #print("[GAT] x_mirna after ReLU shape:", x_mirna.shape)
            # ##print(x_mirna.shape)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            #print("[GAT] x_gene reshaped for conv, shape:", x_gene.shape)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)
            #print("[GAT] x_mirna reshaped for conv, shape:", x_mirna.shape)

            x = torch.cat([x_gene,x_mirna],dim=1)
            #print("[GAT] After concatenating gene and mirna branches, x shape:", x.shape)



        x_parallel = x
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)

        if self.edge_weights:
            x = self.conv1(x, edge_index, edge_weight)

            ## use different activation function based on the models
            x = F.leaky_relu(x, 0.1)
        else:
            # #print('Passing through Conv1 layer without edge_weight.')
            x = self.conv1(x, edge_index)

            ## use different activation function based on the models
            x = F.leaky_relu(x, 0.1)

        if self.edge_weights:
            x = self.conv2(x, edge_index, edge_weight)

            x = F.leaky_relu(x, 0.1)
        else:
            x = self.conv2(x, edge_index) ## output shape: [batches * num_node, hid * head]

            x = F.leaky_relu(x, 0.1)
        ## pooling on the graph to reduce nodes
        x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, hid * head]
        x = self.graph_max_pool(x, self.poolsize)   ## if "gat", then output shape: [batches, floor(num_node / poolsize), hid * head]
                                                        ## if "gcn", then output shape: [batches, floor(num_node / poolsize), hid]

        x = x.view(-1, self.hid * self.head) ## output shape:[batches * floor(num_node / poolsize), hid * head]

        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * hid * head]
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)

        if self.decoder:
            x_reconstruct = x
            x_reconstruct = self.decoder_1(x_reconstruct)
            x_reconstruct = F.relu(x_reconstruct)

            x_reconstruct  = nn.Dropout(self.dropout_rate)(x_reconstruct)
            x_reconstruct = self.decoder_2(x_reconstruct)

        if self.parallel:
            ## the two layer shallow FC network
            x_parallel = self.parallel_linear1(x_parallel)
            x_parallel = F.relu(x_parallel)

            x_parallel = self.parallel_linear2(x_parallel)
            x_parallel = F.relu(x_parallel)

            x = torch.cat((x,x_parallel),1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.classifier(x)

        if self.decoder:
            #return x_reconstruct, F.log_softmax(x, dim=1)
            return x_reconstruct, x
        else:
            #return F.log_softmax(x, dim=1)
            return x
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]

                ## separate mirna from the rest
                x_target_cnv = x_target_cnv[:,:-100]
                x_target_exp = x_target_exp_mirna[:,:-100]
                x_target_mirna = x_target_exp_mirna[:,-100:]
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0
        
        loss2 = nn.CrossEntropyLoss()(y, y_target)
        #loss2 = nn.NLLLoss()(y, y_target)
        loss = 1*loss1 + 1*loss2
        
        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss2_val = 0.2* l2_regularization* l2_loss
            loss += loss2_val
        # #printing loss value for debugging
        # #print("Loss value:", loss.item())
        ##print(f"\n[DEBUG LOSS] Total: {loss.item():.4f} | "
        #      f"Class (CE): {loss2.item():.4f} | "
        #      f"Recon (MSE): {loss1.item():.4f} | "
        #      f"L2 Manuale: {loss2_val:.4f}")
        return loss


class GCN(torch.nn.Module):
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(GCN, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)
    
        if method == 'gcn':
            self.conv1 = ChebConv(self.raised_dimension, self.hid, K=5)
            self.conv2 = ChebConv(self.hid, self.hid, K=5)

        if self.poolsize <= 1:
            if method == 'gcn':
                self.linear_input = (self.num_gene + self.num_mirna) * self.hid
        else:
            if method == 'gcn':
                self.linear_input = math.floor((self.num_gene + self.num_mirna) / self.poolsize) * self.hid

        self.linear1 = nn.Linear(self.linear_input, self.linear_input//4)
        self.linear2 = nn.Linear(self.linear_input//4, self.concate_layer)

        if self.decoder:
            if self.num_features == 1:
                ## Omic mode: Exp, mi, Exp+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene+self.num_mirna)
            elif self.num_features == 2:
                ## omic_mode: Exp+CNV, Exp+CNV+mi
                self.decoder_1 = nn.Linear(self.concate_layer, self.concate_layer*2)
                self.decoder_2 = nn.Linear(self.concate_layer*2, self.num_gene*self.num_features + self.num_mirna)


        if self.parallel:
            #change
            parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

            self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//4)
            self.parallel_linear2 = nn.Linear(parallel_input//4, self.concate_layer)
            self.classifier = nn.Linear(self.concate_layer*2, num_classes)
        else:
            self.classifier = nn.Linear(self.concate_layer, num_classes)

    # Max pooling of size p. Must be a power of 2.
    def graph_max_pool(self, x, p):
        if p > 1:
            #print(f"[DEBUG POOL] Input shape: {x.shape}")
            x = x.permute(0,2,1).contiguous()  # x = B x F x V
            x = nn.MaxPool1d(p)(x)             # B x F x V/p
            x = x.permute(0,2,1).contiguous()  # x = B x V/p x F
            #print(f"[DEBUG POOL] Output shape: {x.shape}")
            return x
        else:
            return x
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):
        batches = x.shape[0]
        num_node = x.shape[1]
        #print("[GCN] Input x shape:", x.shape)

        # Pre-convolution linear transforms
        if self.num_mirna == 0 or self.num_features == 1:
            #print("[GCN] Before pre_conv_linear_gene, x shape:", x.shape)
            x = self.pre_conv_linear_gene(x)
            #print("[GCN] After pre_conv_linear_gene, x shape:", x.shape)

            #print("[GCN] Before ReLU (pre_conv_linear_gene)")
            x = F.relu(x)
            #print("[GCN] After ReLU, x shape:", x.shape)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]
            #print("[GCN] x_exp_mirna shape:", x_exp_mirna.shape)
            #print("[GCN] x_cnv shape:", x_cnv.shape)

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]
            #print(f"[DEBUG FORWARD] x_cnv after slicing shape: {x_cnv.shape}")
            #print(f"[DEBUG FORWARD] x_exp after slicing shape: {x_exp.shape}")

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            #print("[GCN] x_gene after concatenation shape:", x_gene.shape)
            x_gene = x_gene.view(-1,self.num_features)
            #print("[GCN] x_gene after view shape:", x_gene.shape)
            x_mirna = x_exp_mirna[:,-100:]
            x_mirna = torch.flatten(x_mirna)
            #print("[GCN] x_mirna after flatten shape:", x_mirna.shape)
            x_mirna = x_mirna.view(-1, 1)

            #print("[GCN] Before pre_conv_linear_gene (gene branch), x_gene shape:", x_gene.shape)
            x_gene = self.pre_conv_linear_gene(x_gene)
            #print("[GCN] After pre_conv_linear_gene (gene branch), x_gene shape:", x_gene.shape)
            x_gene = F.relu(x_gene)
            #print("[GCN] After ReLU (gene branch), x_gene shape:", x_gene.shape)

            #print("[GCN] Before pre_conv_linear_mirna, x_mirna shape:", x_mirna.shape)
            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            #print("[GCN] After pre_conv_linear_mirna, x_mirna shape:", x_mirna.shape)
            x_mirna = F.relu(x_mirna)
            #print("[GCN] After ReLU (mirna), x_mirna shape:", x_mirna.shape)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            #print("[GCN] x_gene reshaped for conv, shape:", x_gene.shape)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)
            #print("[GCN] x_mirna reshaped for conv, shape:", x_mirna.shape)

            x = torch.cat([x_gene,x_mirna],dim=1)
            #print("[GCN] After concatenating gene and mirna branches, x shape:", x.shape)

        #print("[GCN] x shape before conv layers:", x.shape)

        x_parallel = x
        x = x.view(-1, self.raised_dimension)
        x_parallel = x_parallel.view(batches,-1)
        #print(f"DEBUG: x.size(0)={x.size(0)}, edge_index.max()={edge_index.max().item()}")
        # First convolution
        #print("[GCN] Before conv1, x shape:", x.shape)
        if self.edge_weights:
            x = self.conv1(x, edge_index, edge_weight)
            #print("[GCN] After conv1 (with edge_weights), x shape:", x.shape)
            x = F.relu(x)
            #print("[GCN] After ReLU (conv1), x shape:", x.shape)
        else:
            x = self.conv1(x, edge_index)
            #print("[GCN] After conv1 (no edge_weights), x shape:", x.shape)
            x = F.relu(x)
            #print("[GCN] After ReLU (conv1), x shape:", x.shape)

        # Second convolution
        #print("[GCN] Before conv2, x shape:", x.shape)
        if self.edge_weights:
            x = self.conv2(x, edge_index, edge_weight)
            #print("[GCN] After conv2 (with edge_weights), x shape:", x.shape)
            x = F.relu(x)
            #print("[GCN] After ReLU (conv2), x shape:", x.shape)
        else:
            x = self.conv2(x, edge_index)
            #print("[GCN] After conv2 (no edge_weights), x shape:", x.shape)
            x = F.relu(x)
            #print("[GCN] After ReLU (conv2), x shape:", x.shape)

        #print("[GCN] x shape before pooling:", x.shape)

        ## pooling on the graph to reduce nodes
        x = x.view(batches, num_node, -1) ## output shape: [batches, num_node, hid * head]
        x = self.graph_max_pool(x, self.poolsize)   ## if "gat", then output shape: [batches, floor(num_node / poolsize), hid * head]
                                                    ## if "gcn", then output shape: [batches, floor(num_node / poolsize), hid]
        #print("[GCN] x shape after pooling:", x.shape)
        if self.method == 'gcn':
            x = x.view(-1, self.hid) ## output shape:[batches * floor(num_node / poolsize), hid]
            #print("[GCN] After view for gcn method, x shape:", x.shape)

        x = x.view(batches, -1) ## output size: [batches, floor(num_node / poolsize) * hid * head]
        #print("[GCN] Before linear1, x shape:", x.shape)
        x = self.linear1(x)
        #print("[GCN] After linear1, x shape:", x.shape)
        x = F.relu(x)
        #print("[GCN] After ReLU (linear1), x shape:", x.shape)
        #print("[GCN] Before linear2, x shape:", x.shape)
        x = self.linear2(x)
        #print("[GCN] After linear2, x shape:", x.shape)
        x = F.relu(x)
        #print("[GCN] After ReLU (linear2), x shape:", x.shape)

        if self.decoder:
            x_reconstruct = x
            #print("[GCN] Before decoder_1, x_reconstruct shape:", x_reconstruct.shape)
            x_reconstruct = self.decoder_1(x_reconstruct)
            #print("[GCN] After decoder_1, x_reconstruct shape:", x_reconstruct.shape)
            x_reconstruct = F.relu(x_reconstruct)
            #print("[GCN] After ReLU (decoder_1), x_reconstruct shape:", x_reconstruct.shape)

            x_reconstruct  = nn.Dropout(0.2)(x_reconstruct)
            #print("[GCN] After Dropout (decoder_1), x_reconstruct shape:", x_reconstruct.shape)
            x_reconstruct = self.decoder_2(x_reconstruct)
            #print("[GCN] After decoder_2, x_reconstruct shape:", x_reconstruct.shape)

        if self.parallel:
            ## the two layer shallow FC network
            #print("[GCN] Before parallel_linear1, x_parallel shape:", x_parallel.shape)
            x_parallel = self.parallel_linear1(x_parallel)
            #print("[GCN] After parallel_linear1, x_parallel shape:", x_parallel.shape)
            #x_parallel = F.relu(x_parallel)
            x_parallel = F.leaky_relu(x_parallel, 0.1)
            #print("[GCN] After ReLU (parallel_linear1), x_parallel shape:", x_parallel.shape)

            #print("[GCN] Before parallel_linear2, x_parallel shape:", x_parallel.shape)
            x_parallel = self.parallel_linear2(x_parallel)
            ##print("[GCN] After parallel_linear2, x_parallel shape:", x_parallel.shape)
            #x_parallel = F.relu(x_parallel)
            x_parallel = F.leaky_relu(x_parallel, 0.1)
            #print("[GCN] After ReLU (parallel_linear2), x_parallel shape:", x_parallel.shape)

            x = torch.cat((x,x_parallel),1)
            #print("[GCN] After concatenating parallel branch, x shape:", x.shape)
        #print("[GCN] Before Dropout, x shape:", x.shape)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        #print("[GCN] After Dropout, x shape:", x.shape)
        #print("[GCN] Before classifier, x shape:", x.shape)
        x = self.classifier(x)
        #print("[GCN] After classifier, x shape:", x.shape)

        if self.decoder:
            return x_reconstruct, F.log_softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        if self.decoder:
            if self.num_mirna == 0 or self.num_features == 1:
                x_target = x_target.view(x_target.size()[0], -1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target)
            else:
                x_target_exp_mirna = x_target[:,:,0]
                x_target_cnv = x_target[:,:,1]

                ## separate mirna from the rest
                x_target_cnv = x_target_cnv[:,:-100]
                x_target_exp = x_target_exp_mirna[:,:-100]
                x_target_mirna = x_target_exp_mirna[:,-100:]
                x_target_flatten = torch.cat([x_target_exp, x_target_cnv, x_target_mirna], dim=1)
                loss1 = nn.MSELoss()(x_reconstruct, x_target_flatten)
        else:
            loss1 = 0
        # Da controllare il fatto log_softmax + CrossEntropyLoss
        #loss2 = nn.CrossEntropyLoss()(y, y_target)
        loss2 = nn.NLLLoss()(y, y_target)
        loss = 1*loss1 + 1*loss2
        
        if self.l2:
            l2_loss = 0.0
            for param in self.parameters():
                data = param* param
                l2_loss += data.sum()

            loss += 0.2* l2_regularization* l2_loss
        return loss


class Baseline(torch.nn.Module):
    def __init__(self, 
                    method, 
                    parallel, 
                    l2, 
                    decoder, 
                    poolsize, 
                    poolrate,
                    edge_weights, 
                    edge_attributes, 
                    num_gene,
                    num_mirna, 
                    omic_mode, 
                    num_classes, 
                    dropout_rate):

        super(Baseline, self).__init__()
        self.omic_mode = omic_mode
        self.method = method
        self.parallel = parallel
        self.decoder = decoder
        self.l2 = l2
        self.poolsize = poolsize
        self.poolrate = poolrate
        self.edge_weights = edge_weights
        self.edge_attributes = edge_attributes
        self.hid = 6
        self.num_gene = num_gene
        self.num_mirna = num_mirna
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.raised_dimension = 8
        self.concate_layer = 64

        if self.omic_mode < 3:
            self.num_features = 1
        else:
            self.num_features = 2

        self.pre_conv_linear_gene = nn.Linear(self.num_features, self.raised_dimension)
        self.pre_conv_linear_mirna = nn.Linear(1, self.raised_dimension)

        parallel_input = self.raised_dimension*(self.num_gene + self.num_mirna)

        self.parallel_linear1 = nn.Linear(parallel_input, parallel_input//2)
        self.parallel_linear2 = nn.Linear(parallel_input//2, parallel_input//4)
        self.parallel_linear3 = nn.Linear(parallel_input//4, self.concate_layer)
        self.classifier = nn.Linear(self.concate_layer, num_classes)
    
    ## create the batch index for each nodes in the batch
    def create_batch_index(self, batches):
        batch_index = []
        for i in range(batches):
            batch_index += [i]*(self.num_gene + self.num_mirna)
        return(torch.Tensor(batch_index).type(torch.int64))
        
    def forward(self, x, edge_index, edge_weight):

        batches = x.shape[0]
        num_node = x.shape[1]
        
        if self.num_mirna == 0 or self.num_features == 1:
            x = self.pre_conv_linear_gene(x)
            x = F.relu(x)
        else:
            ## the second matrix cnv_data has padding
            x_exp_mirna = x[:,:,0]
            x_cnv = x[:,:,1]

            ## separate mirna from the rest
            x_cnv = x_cnv[:,:-100]
            x_exp = x_exp_mirna[:,:-100]

            x_cnv = x_cnv.view(batches,-1,1)
            x_exp = x_exp.view(batches,-1,1)
            x_gene = torch.cat([x_exp,x_cnv],dim=1)
            x_gene = x_gene.view(-1,self.num_features)
            x_mirna = x_exp_mirna[:,-100:]
            x_mirna = torch.flatten(x_mirna)
            x_mirna = x_mirna.view(-1, 1)

            x_gene = self.pre_conv_linear_gene(x_gene)
            x_gene = F.relu(x_gene)

            x_mirna = self.pre_conv_linear_mirna(x_mirna)
            x_mirna = F.relu(x_mirna)

            x_gene = x_gene.view(batches, -1, self.raised_dimension)
            x_mirna = x_mirna.view(batches, -1, self.raised_dimension)

            x = torch.cat([x_gene,x_mirna],dim=1)



        x_parallel = x
        x_parallel = x_parallel.view(batches,-1)
        
        x_parallel = self.parallel_linear1(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear2(x_parallel)
        x_parallel = F.relu(x_parallel)
        x_parallel = self.parallel_linear3(x_parallel)
        x_parallel = F.relu(x_parallel)

        x_parallel = F.dropout(x_parallel, p=self.dropout_rate, training=self.training)
        x_parallel = self.classifier(x_parallel)
        return F.log_softmax(x_parallel, dim=1)
    
    def loss(self, x_reconstruct, x_target, y, y_target, l2_regularization):
        loss2 = nn.CrossEntropyLoss()(y, y_target)
        loss = 1*loss2
        
        return loss