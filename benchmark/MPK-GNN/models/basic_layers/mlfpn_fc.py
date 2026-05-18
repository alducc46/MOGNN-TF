'''
@Project: MPK-GNN
@File   : mlfpn_fc.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc   : Implementazione di Multi-Layer Forward Propagation Network con Fully Connected layers
    
    Questa classe definisce una rete neurale multi-layer dove ogni layer è completamente
    connesso (fully connected). È un componente riutilizzabile utilizzato in diversi
    moduli di MPK-GNN come Sample Learning Module e Projection Module.
    
    Caratteristiche:
    1. Numero arbitrario di layer configurabile tramite lista dimensioni
    2. Funzione di attivazione applicata a tutti i layer tranne l'ultimo
    3. Costruzione automatica della rete tramite utility functions
    4. Supporto per diverse funzioni di attivazione (default: ReLU)
    
    Esempio utilizzo:
    - Input: [batch_size, 1000] 
    - Hidden layers: [512, 256]
    - Output: [batch_size, 128]
    - dims = [1000, 512, 256, 128]
'''
import torch
import torch.nn as nn
from typing import List

# Importa utility per costruzione automatica dei layer
from utils.ops_al import build_layer_units


class MLFPN_FC(nn.Module):
    """
    Multi-Layer Forward Propagation Network con Fully Connected layers.
    
    Implementa una rete neurale feed-forward completamente connessa con
    numero arbitrario di layer. Ogni layer nascosto è seguito da una
    funzione di attivazione, mentre l'ultimo layer non ha attivazione
    (per permettere flessibilità nell'uso downstream).
    
    Args:
        dims (List[int]): Lista delle dimensioni dei layer. 
                          Esempio: [input_dim, hidden1_dim, hidden2_dim, output_dim]
        act_func (nn.Module): Funzione di attivazione applicata ai layer nascosti.
                              Default: ReLU()
    
    Esempio:
        # Rete con input 1000, hidden 512, output 256
        net = MLFPN_FC([1000, 512, 256], nn.ReLU())
        # Input: [batch_size, 1000] -> Output: [batch_size, 256]
    """

    def __init__(self, dims: List[int], act_func: nn.Module = nn.ReLU()):
        super(MLFPN_FC, self).__init__()

        # Costruisce automaticamente la rete utilizzando utility function
        # 'linear' specifica che vogliamo layer fully connected
        # dims specifica le dimensioni di input/output di ogni layer
        # act_func viene applicata a tutti i layer tranne l'ultimo
        self.network = build_layer_units(layer_type='linear', dims=dims, act_func=act_func)

    def forward(self, fea: torch.Tensor) -> torch.Tensor:
        """
        Forward pass attraverso la rete multi-layer.
        
        Applica sequenzialmente tutti i layer della rete:
        1. Per ogni layer nascosto: Linear -> Activation
        2. Per l'ultimo layer: solo Linear (no activation)
        
        Args:
            fea (torch.Tensor): Matrice delle feature di input
                               Shape: [batch_size, input_dim]
        
        Returns:
            torch.Tensor: Matrice delle feature elaborate
                         Shape: [batch_size, output_dim]
        """
        output = fea
        
        # Itera attraverso tutti i layer della rete
        for seq in self.network:
            if len(seq) == 1:  
                # Ultimo layer: solo trasformazione lineare senza attivazione
                output = seq[0](output)
            elif len(seq) == 2:  
                # Layer nascosto: trasformazione lineare + funzione di attivazione
                output = seq[0](output)  # Layer lineare
                output = seq[1](output)  # Funzione di attivazione
                
        return output