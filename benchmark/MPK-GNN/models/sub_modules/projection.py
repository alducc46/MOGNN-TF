'''
@Project: MPK-GNN
@File   : projection.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc   : Implementazione del Projection Module (PM)
    
    Il Projection Module è progettato per il contrastive learning.
    Prende le rappresentazioni dal Feature Learning Module e le proietta
    in uno spazio di dimensionalità inferiore ottimizzato per il calcolo
    della supervised contrastive loss.
    
    Questo modulo aiuta il modello a imparare rappresentazioni che:
    1. Portano campioni della stessa classe più vicini
    2. Allontanano campioni di classi diverse
    3. Catturano caratteristiche discriminative importanti
    
    Architettura:
    Input: [batch_size, flm_fl_dim] (da Feature Learning Module)
    ↓
    Fully Connected Layers con ReLU activation
    ↓
    Output: [batch_size, pm_dim_2] (rappresentazione proiettata)
'''
import torch.nn as nn

# Importa la classe per reti fully connected multi-layer
from models.basic_layers.mlfpn_fc import MLFPN_FC


class Projection(nn.Module):
    """
    Projection Module (PM) per contrastive learning in MPK-GNN.
    
    Questo modulo proietta le rappresentazioni dal Feature Learning Module
    in uno spazio ottimizzato per contrastive learning. La proiezione aiuta
    il modello a imparare rappresentazioni discriminative che separano meglio
    le diverse classi nel spazio latente.
    
    Args:
        flm_fl_o (int): Dimensione input (output del Feature Learning Module)
        pm_dim_1 (int): Dimensione del primo layer nascosto
        pm_dim_2 (int): Dimensione output del modulo (spazio di proiezione)
    """
    
    def __init__(self, flm_fl_o, pm_dim_1, pm_dim_2):
        super(Projection, self).__init__()

        # Definisce le dimensioni della rete di proiezione
        # Input: flm_fl_o (da Feature Learning Module)
        # Hidden: pm_dim_1  
        # Output: pm_dim_2 (spazio di proiezione per contrastive learning)
        pm_dims = [flm_fl_o, pm_dim_1, pm_dim_2]
        
        # Crea la rete fully connected con attivazione ReLU
        # Questa rete trasforma le rappresentazioni in uno spazio ottimale
        # per il calcolo della contrastive loss
        self.network = MLFPN_FC(pm_dims, nn.ReLU())

    def forward(self, feat):
        """
        Forward pass del Projection Module.
        
        Args:
            feat (torch.Tensor): Features dal Feature Learning Module 
                                 con shape [batch_size, flm_fl_o]
        
        Returns:
            torch.Tensor: Rappresentazione proiettata con shape [batch_size, pm_dim_2]
                         Ottimizzata per contrastive learning
        """
        return self.network(feat)

