'''
@Project: MPK-GNN
@File   : sample_learning.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc   : Implementazione del Sample-based Learning Module (SLM)
    
    Il Sample Learning Module elabora i dati multi-omici a livello di campione.
    Prende in input i dati concatenati (flattened) e li elabora attraverso
    una rete fully connected multi-layer.
    
    Questo modulo cattura le relazioni tra diversi tipi di dati omici
    (es. gene expression + CNV) per ogni singolo campione.
    
    Architettura:
    Input: [batch_size, num_genes * num_omics] (dati concatenati)
    ↓
    Fully Connected Layers con ReLU activation
    ↓
    Output: [batch_size, slm_dim_2] (rappresentazione del campione)
'''
import torch.nn as nn

# Importa la classe per reti fully connected multi-layer
from models.basic_layers.mlfpn_fc import MLFPN_FC


class SampleLearning(nn.Module):
    """
    Sample-based Learning Module (SLM) per MPK-GNN.
    
    Questo modulo elabora i dati multi-omici a livello di campione utilizzando
    fully connected layers. È progettato per catturare le interazioni complesse
    tra diversi tipi di dati omici (es. gene expression e CNV) all'interno
    di ogni singolo campione.
    
    Args:
        num_omics (int): Numero di tipi di dati omici (es. 2 per expression+CNV)
        num_genes (int): Numero di geni nel dataset
        slm_dim_1 (int): Dimensione del primo layer nascosto
        slm_dim_2 (int): Dimensione del secondo layer nascosto (output del modulo)
    """

    def __init__(self, num_omics, num_genes, slm_dim_1, slm_dim_2):
        super(SampleLearning, self).__init__()

        # Definisce le dimensioni della rete multi-layer
        # Input: num_omics * num_genes (dati concatenati)
        # Hidden: slm_dim_1
        # Output: slm_dim_2
        slm_dims = [num_omics * num_genes, slm_dim_1, slm_dim_2]
        
        # Crea la rete fully connected con funzione di attivazione ReLU
        # MLFPN_FC gestisce automaticamente la creazione di layer multipli
        self.network = MLFPN_FC(slm_dims, nn.ReLU())

    def forward(self, feat):
        """
        Forward pass del Sample Learning Module.
        
        Args:
            feat (torch.Tensor): Features di input con shape [batch_size, num_genes * num_omics]
                                 Rappresenta i dati multi-omici concatenati per ogni campione
        
        Returns:
            torch.Tensor: Rappresentazione elaborata con shape [batch_size, slm_dim_2]
                         Codifica le caratteristiche a livello di campione
        """
        return self.network(feat)
