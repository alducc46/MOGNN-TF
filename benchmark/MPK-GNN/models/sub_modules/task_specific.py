'''
@Project: MPK-GNN
@File   : task_specific.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc   : Implementazione del Task-Specific Module (TSM)
    
    Il Task-Specific Module è il layer finale di classificazione del modello MPK-GNN.
    Combina le rappresentazioni dal Sample Learning Module (SLM) e dal Feature Learning
    Module (FLM) per produrre le predizioni finali di classificazione.
    
    Questo modulo:
    1. Concatena output di SLM e FLM (da 3 diverse matrici di adiacenza)
    2. Applica una trasformazione lineare per la classificazione
    3. Applica log_softmax per ottenere probabilità logaritmiche
    
    Architettura:
    Input: Concatenazione di [SLM_output, FLM_output_1, FLM_output_2, FLM_output_3]
           Shape: [batch_size, slm_dim_2 + flm_fl_dim * 3]
    ↓
    Linear Layer: → [num_classes]
    ↓
    Log Softmax: → probabilità logaritmiche
    ↓
    Output: [batch_size, num_classes] (logits per classificazione)
'''
import torch.nn as nn
import torch.nn.functional as F


class TaskSpecific(nn.Module):
    """
    Task-Specific Module (TSM) per classificazione finale in MPK-GNN.
    
    Questo è il modulo di output finale che combina le rappresentazioni
    apprese dai moduli Sample Learning (SLM) e Feature Learning (FLM)
    per produrre le predizioni di classificazione.
    
    Il modulo fonde informazioni complementari:
    - SLM: cattura pattern a livello di campione (sample-level)
    - FLM: cattura relazioni strutturali tra geni (feature-level) 
           da tre diverse matrici di adiacenza
    
    Args:
        slm_dim_2 (int): Dimensione output del Sample Learning Module
        flm_fl_dim (int): Dimensione output del Feature Learning Module
        num_classes (int): Numero di classi per la classificazione finale
    """

    def __init__(self, slm_dim_2, flm_fl_dim, num_classes):
        super(TaskSpecific, self).__init__()
        
        # Layer lineare finale per classificazione
        # Input: concatenazione di [SLM_output, FLM_output_1, FLM_output_2, FLM_output_3]
        # Dimensione input: slm_dim_2 + (flm_fl_dim * 3)
        # Output: num_classes (logits per ogni classe)
        self.network = nn.Linear(slm_dim_2 + flm_fl_dim * 3, num_classes)

    def forward(self, h):
        """
        Forward pass del Task-Specific Module.
        
        Args:
            h (torch.Tensor): Features concatenate con shape [batch_size, slm_dim_2 + flm_fl_dim * 3]
                             Combina output di SLM e FLM (da 3 matrici di adiacenza)
        
        Returns:
            torch.Tensor: Logits di classificazione con shape [batch_size, num_classes]
                         Probabilità logaritmiche per ogni classe
        """
        
        # Applica trasformazione lineare finale
        h = self.network(h)  # [batch_size, num_classes]
        
        # Applica log_softmax per ottenere probabilità logaritmiche
        # Questo è compatibile con CrossEntropyLoss che si aspetta logits raw
        h = F.log_softmax(h, dim=1)  # [batch_size, num_classes]

        return h
