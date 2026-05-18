'''
@Project: MPK-GNN
@File   : mpk.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc   : Implementazione del framework MPK-GNN proposto
    
    Questo file contiene la classe principale MPK che implementa l'architettura
    MPK-GNN (Multi-Prior Knowledge Graph Neural Network).
    
    L'architettura è composta da quattro moduli principali:
    1. Sample Learning Module (SLM): Elabora i dati tramite fully connected layers
    2. Feature Learning Module (FLM): Elabora i dati tramite Graph Convolutional Networks
    3. Projection Module (PM): Proietta le features per contrastive learning
    4. Task-Specific Module (TSM): Layer finale di classificazione
    
    Il modello utilizza tre diverse matrici di adiacenza per catturare
    diversi tipi di conoscenza biologica a priori (es. PPI, co-espressione, pathway).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importa i quattro moduli principali dell'architettura MPK-GNN
from models.sub_modules.projection import Projection
from models.sub_modules.task_specific import TaskSpecific
from models.sub_modules.sample_learning import SampleLearning
from models.sub_modules.feature_learning import FeatureLearning


class MPK(nn.Module):
    """
    Classe principale del modello MPK-GNN (Multi-Prior Knowledge Graph Neural Network).
    
    Questo modello combina informazioni a livello di campione e a livello di feature
    utilizzando sia fully connected networks che graph neural networks.
    Supporta l'integrazione di multiple fonti di conoscenza biologica a priori
    attraverso diverse matrici di adiacenza.
    
    Args:
        num_omics (int): Numero di tipi di dati omici (es. 2 per expression+CNV)
        num_genes (int): Numero di geni nel dataset
        num_classes (int): Numero di classi per la classificazione
        slm_dim_1 (int): Dimensione primo layer del Sample Learning Module
        slm_dim_2 (int): Dimensione secondo layer del Sample Learning Module
        flm_gcn_dim_1 (int): Dimensione primo layer GCN del Feature Learning Module
        flm_gcn_dim_2 (int): Dimensione secondo layer GCN del Feature Learning Module
        pool_size (int): Dimensione del pooling layer (deve essere potenza di 2)
        flm_fl_dim (int): Dimensione del flatten layer nel Feature Learning Module
        pm_dim_1 (int): Prima dimensione del Projection Module
        pm_dim_2 (int): Seconda dimensione del Projection Module
    """

    def __init__(self, num_omics, num_genes, num_classes, slm_dim_1, slm_dim_2, flm_gcn_dim_1,
                 flm_gcn_dim_2, pool_size, flm_fl_dim, pm_dim_1, pm_dim_2):
        super(MPK, self).__init__()

        # === DEFINIZIONE MODULI DELL'ARCHITETTURA ===
        
        # Sample-based Learning Module: elabora i dati concatenati (flattened)
        # Input: [batch_size, num_genes * num_omics] -> Output: [batch_size, slm_dim_2]
        self.slm = SampleLearning(num_omics, num_genes, slm_dim_1, slm_dim_2)

        # Feature-based Learning Module: elabora i dati tramite GCN
        # Input: [batch_size, num_genes, num_omics] -> Output: [batch_size, flm_fl_dim]
        self.flm = FeatureLearning(num_omics, num_genes, flm_gcn_dim_1, flm_gcn_dim_2, pool_size, flm_fl_dim)

        # Projection Module: proietta le feature per contrastive learning
        # Input: [batch_size, flm_fl_dim] -> Output: [batch_size, pm_dim_2]
        self.pm = Projection(flm_fl_dim, pm_dim_1, pm_dim_2)

        # Task-specific Module: layer finale di classificazione
        # Input: [batch_size, slm_dim_2 + flm_fl_dim * 3] -> Output: [batch_size, num_classes]
        self.tsm = TaskSpecific(slm_dim_2, flm_fl_dim, num_classes)

    def forward(self, x_in, adj_0, adj_1, adj_2):
        """
        Forward pass del modello MPK-GNN.
        
        Il modello elabora i dati attraverso due pathway paralleli:
        1. Sample-based pathway: utilizza fully connected layers
        2. Feature-based pathway: utilizza graph convolutional networks
        
        Args:
            x_in (torch.Tensor): Matrice di input con shape [batch_size, num_genes, num_omics]
                                 Contiene i dati multi-omici (es. expression + CNV)
            adj_0 (torch.sparse.FloatTensor): Prima matrice di adiacenza [num_genes, num_genes]
                                              (es. rete protein-protein interaction)
            adj_1 (torch.sparse.FloatTensor): Seconda matrice di adiacenza [num_genes, num_genes]
                                              (es. rete di co-espressione genica)
            adj_2 (torch.sparse.FloatTensor): Terza matrice di adiacenza [num_genes, num_genes]
                                              (es. rete di pathway biologici)
        
        Returns:
            tuple: (output_classificazione, repr_proj_0, repr_proj_1, repr_proj_2)
                - output_classificazione: logits per classificazione [batch_size, num_classes]
                - repr_proj_0/1/2: rappresentazioni proiettate per contrastive learning [batch_size, pm_dim_2]
        """
        
        # === PREPARAZIONE INPUT ===
        x = x_in  # Input per il Feature Learning Module: [batch_size, num_genes, num_omics]
        x_nn = x_in.view(x_in.size()[0], -1)  # Input per il Sample Learning Module: [batch_size, num_genes * num_omics]

        # === SAMPLE-BASED LEARNING MODULE (SLM) ===
        # Elabora i dati concatenati tramite fully connected layers
        o_slm = self.slm(x_nn)  # Output: [batch_size, slm_dim_2]

        # === FEATURE-BASED LEARNING MODULE (FLM) ===
        # Elabora i dati tramite GCN utilizzando le tre diverse matrici di adiacenza
        # Questo permette di catturare diversi tipi di relazioni biologiche
        o_flm_0 = self.flm(x, adj_0)  # Output con prima matrice: [batch_size, flm_fl_dim]
        o_flm_1 = self.flm(x, adj_1)  # Output con seconda matrice: [batch_size, flm_fl_dim]
        o_flm_2 = self.flm(x, adj_2)  # Output con terza matrice: [batch_size, flm_fl_dim]
        
        # Concatena gli output dei tre FLM per il modulo task-specific
        o_flm = torch.cat((o_flm_0, o_flm_1, o_flm_2), 1)  # [batch_size, flm_fl_dim * 3]

        # === PROJECTION MODULE (PM) ===
        # Proietta le rappresentazioni per contrastive learning
        # Ogni rappresentazione viene proiettata separatamente
        o_pm_0 = self.pm(o_flm_0)  # [batch_size, pm_dim_2]
        o_pm_1 = self.pm(o_flm_1)  # [batch_size, pm_dim_2]
        o_pm_2 = self.pm(o_flm_2)  # [batch_size, pm_dim_2]

        # === TASK-SPECIFIC MODULE (TSM) ===
        # Combina output di SLM e FLM per la classificazione finale
        i_tsm = torch.cat((o_flm, o_slm), 1)  # [batch_size, slm_dim_2 + flm_fl_dim * 3]
        o_tsm = self.tsm(i_tsm)               # [batch_size, num_classes]

        # Restituisce output di classificazione e rappresentazioni normalizzate per contrastive learning
        return o_tsm, F.normalize(o_pm_0, dim=1), F.normalize(o_pm_1, dim=1), F.normalize(o_pm_2, dim=1)
