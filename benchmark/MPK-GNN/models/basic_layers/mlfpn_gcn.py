'''
@Project: MPK-GNN
@File   : mlfpn_gcn.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc   : Implementazione di Multi-Layer Forward Propagation Network con Graph Convolution layers
    
    Questa classe definisce una rete neurale multi-layer dove ogni layer è una
    operazione di convoluzione su grafo (Graph Convolution). È progettata per
    elaborare dati strutturati come grafi, utilizzando matrici di adiacenza
    per catturare relazioni tra nodi.
    
    A differenza di MLFPN_FC che opera su dati tabulari, MLFPN_GCN:
    1. Richiede una matrice di adiacenza per definire la struttura del grafo
    2. Applica operazioni di aggregazione dai nodi vicini
    3. È ottimizzata per catturare relazioni strutturali nei dati
    
    Utilizzata principalmente nel Feature Learning Module per incorporare
    conoscenza biologica a priori (es. protein-protein interactions).
    
    Esempio utilizzo:
    - Nodi: geni (features: expression + CNV)
    - Edges: interazioni protein-protein 
    - Layer: trasformano features dei geni aggregando da vicini
'''
import torch
import torch.nn as nn
from typing import List

# Importa utility per costruzione automatica dei layer
from utils.ops_al import build_layer_units


class MLFPN_GCN(nn.Module):
    """
    Multi-Layer Forward Propagation Network con Graph Convolution layers.
    
    Implementa una rete neurale multi-layer dove ogni layer è una operazione
    di convoluzione su grafo. Utilizzata per elaborare dati con struttura a
    grafo, come reti biologiche di interazione tra geni/proteine.
    
    Ogni layer GCN:
    1. Aggrega informazioni dai nodi vicini secondo la matrice di adiacenza
    2. Applica trasformazione lineare
    3. Applica funzione di attivazione (tranne ultimo layer)
    
    Args:
        dims (List[int]): Lista delle dimensioni dei layer. 
                          Esempio: [input_features, hidden_dim, output_dim]
        act_func (nn.Module): Funzione di attivazione per layer nascosti.
                              Default: ReLU()
    
    Esempio:
        # GCN con 2 features per nodo, hidden 64, output 32
        gcn = MLFPN_GCN([2, 64, 32], nn.ReLU())
        # Input: features [num_nodes, 2], adjacency [num_nodes, num_nodes]
        # Output: [num_nodes, 32]
    """

    def __init__(self, dims: List[int], act_func: nn.Module=nn.ReLU()):
        super(MLFPN_GCN, self).__init__()

        # Costruisce automaticamente la rete GCN multi-layer
        # 'gcn' specifica che vogliamo layer di graph convolution
        # dims specifica le dimensioni delle features per ogni layer
        # act_func viene applicata a tutti i layer tranne l'ultimo
        self.network = build_layer_units(layer_type='gcn', dims=dims, act_func=act_func)

    def forward(self, fea: torch.Tensor, adj: torch.sparse) -> torch.Tensor:
        """
        Forward pass attraverso la rete GCN multi-layer.
        
        Applica sequenzialmente tutti i layer GCN:
        1. Per ogni layer nascosto: GCN -> Activation
        2. Per l'ultimo layer: solo GCN (no activation)
        
        Args:
            fea (torch.Tensor): Matrice delle feature dei nodi
                               Shape: [num_nodes, input_features]
            adj (torch.sparse): Matrice di adiacenza sparsa del grafo
                               Shape: [num_nodes, num_nodes]
        
        Returns:
            torch.Tensor: Feature elaborate dei nodi
                         Shape: [num_nodes, output_features]
        """
        output = fea
        
        # Itera attraverso tutti i layer GCN
        for seq in self.network:
            if len(seq) == 1:  
                # Ultimo layer: solo graph convolution senza attivazione
                output = seq[0](output, adj)
            elif len(seq) == 2:  
                # Layer nascosto: graph convolution + funzione di attivazione
                output = seq[0](output, adj)  # Layer GCN
                output = seq[1](output)       # Funzione di attivazione
                
        return output