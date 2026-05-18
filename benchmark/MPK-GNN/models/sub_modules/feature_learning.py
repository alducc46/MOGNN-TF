'''
@Project: MPK-GNN
@File   : feature_learning.py
@Author : Shunxin Xiao
@Email  : xiaoshunxin.tj@gmail
@Desc   : Implementazione del Feature-based Learning Module (FLM)
    
    Il Feature Learning Module elabora i dati multi-omici a livello di feature
    utilizzando Graph Convolutional Networks (GCN). Questo modulo è composto da:
    1. Multi-layer Graph Convolutional Network (2 layer)
    2. Graph Max Pooling layer per riduzione dimensionalità 
    3. Flatten layer per preparare l'output
    
    Il modulo utilizza matrici di adiacenza per catturare relazioni biologiche
    tra i geni (es. protein-protein interactions, co-espressione, pathway).
    
    Architettura:
    Input: [batch_size, num_genes, num_omics] + adjacency_matrix
    ↓
    GCN Layer 1: [num_omics] → [flm_gcn_dim_1] con ReLU
    ↓  
    GCN Layer 2: [flm_gcn_dim_1] → [flm_gcn_dim_2] con ReLU
    ↓
    Graph Max Pooling: riduce num_genes di un fattore pool_size
    ↓
    Flatten Layer: → [flm_fl_dim]
    ↓
    Output: [batch_size, flm_fl_dim]
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureLearning(nn.Module):
    """
    Feature-based Learning Module (FLM) per MPK-GNN.
    
    Questo modulo elabora i dati multi-omici a livello di feature utilizzando
    Graph Convolutional Networks per catturare le relazioni strutturali
    tra i geni basate su conoscenza biologica a priori.
    
    Args:
        num_omics (int): Numero di tipi di dati omici
        num_genes (int): Numero di geni
        flm_gcn_dim_1 (int): Dimensione output del primo layer GCN
        flm_gcn_dim_2 (int): Dimensione output del secondo layer GCN  
        pool_size (int): Fattore di riduzione per graph pooling (deve essere potenza di 2)
        flm_fl_dim (int): Dimensione finale dopo flatten layer
    """

    def __init__(self, num_omics, num_genes, flm_gcn_dim_1, flm_gcn_dim_2, pool_size, flm_fl_dim):
        super(FeatureLearning, self).__init__()
        
        # Salva parametri per uso nei metodi
        self.num_omics = num_omics
        self.num_genes = num_genes
        self.flm_gcn_dim_1 = flm_gcn_dim_1
        self.flm_gcn_dim_2 = flm_gcn_dim_2
        self.pool_size = pool_size

        # === DEFINIZIONE LAYER GCN ===
        
        # Primo layer GCN: trasforma da num_omics a flm_gcn_dim_1 dimensioni
        self.gcn_1 = nn.Linear(self.num_omics, self.flm_gcn_dim_1)

        # Secondo layer GCN: trasforma da flm_gcn_dim_1 a flm_gcn_dim_2 dimensioni
        self.gcn_2 = nn.Linear(self.flm_gcn_dim_1, flm_gcn_dim_2)

        # === FLATTEN LAYER ===
        
        # Calcola dimensione input del flatten layer dopo pooling
        self.fl_dim_i = flm_gcn_dim_2 * (num_genes // pool_size)
        
        # Layer finale che trasforma in rappresentazione finale
        self.fl = nn.Linear(self.fl_dim_i, flm_fl_dim)

    def graph_conv_net(self, feat, adj):
        """
        Implementa la rete di convoluzione su grafo a due layer.
        
        Questo metodo applica due layer di convoluzione su grafo consecutivi,
        ognuno seguito da attivazione ReLU. La convoluzione su grafo aggrega
        le informazioni dai nodi vicini secondo la matrice di adiacenza.
        
        Args:
            feat (torch.Tensor): Matrice delle feature con shape [batch_size, num_genes, num_omics]
            adj (torch.sparse.FloatTensor): Matrice di adiacenza con shape [num_genes, num_genes]
        
        Returns:
            torch.Tensor: Feature elaborate con shape [batch_size, num_genes, flm_gcn_dim_2]
        """
        
        # Ottieni le dimensioni del batch
        batch_size, num_genes, num_omics = feat.size()
        
        # === PRIMO LAYER GCN ===
        
        # Trasforma per l'input del primo layer GCN
        # [batch_size, num_genes, num_omics] → [num_genes, num_omics, batch_size]
        x = feat.permute(1, 2, 0).contiguous()
        # [num_genes, num_omics, batch_size] → [num_genes, num_omics * batch_size]
        x = x.view([num_genes, num_omics * batch_size])

        # Applica convoluzione su grafo: moltiplica matrice di adiacenza per features
        # Questo aggrega le informazioni dai nodi connessi nel grafo
        x = torch.mm(adj, x)  # [num_genes, num_omics * batch_size]
        
        # Ripristina la forma originale
        x = x.view([num_genes, num_omics, batch_size])
        # [num_genes, num_omics, batch_size] → [batch_size, num_genes, num_omics]
        x = x.permute(2, 0, 1).contiguous()
        
        # Applica il primo layer lineare e attivazione ReLU
        x = x.view([batch_size * num_genes, num_omics])  # Flatten per applicare linear layer
        x = self.gcn_1(x)  # [batch_size * num_genes, flm_gcn_dim_1]
        x = F.relu(x)      # Attivazione ReLU

        # === SECONDO LAYER GCN ===
        
        # Trasforma per l'input del secondo layer GCN
        # [batch_size * num_genes, flm_gcn_dim_1] → [batch_size, num_genes, flm_gcn_dim_1]
        x = x.view([batch_size, num_genes, self.flm_gcn_dim_1])
        # [batch_size, num_genes, flm_gcn_dim_1] → [num_genes, flm_gcn_dim_1, batch_size]
        x = x.permute(1, 2, 0).contiguous()
        # [num_genes, flm_gcn_dim_1, batch_size] → [num_genes, flm_gcn_dim_1 * batch_size]
        x = x.view([num_genes, self.flm_gcn_dim_1 * batch_size])

        # Applica la seconda convoluzione su grafo
        x = torch.mm(adj, x)  # [num_genes, flm_gcn_dim_1 * batch_size]
        
        # Ripristina la forma per il secondo layer lineare
        x = x.view([num_genes, self.flm_gcn_dim_1, batch_size])
        # [num_genes, flm_gcn_dim_1, batch_size] → [batch_size, num_genes, flm_gcn_dim_1]
        x = x.permute(2, 0, 1).contiguous()
        
        # Applica il secondo layer lineare e attivazione ReLU
        x = x.view([batch_size * num_genes, self.flm_gcn_dim_1])  # Flatten
        x = self.gcn_2(x)  # [batch_size * num_genes, flm_gcn_dim_2]
        x = F.relu(x)      # Attivazione ReLU

        # Output finale del GCN multi-layer
        x = x.view([batch_size, num_genes, self.flm_gcn_dim_2])  # [batch_size, num_genes, flm_gcn_dim_2]
        
        return x

    def graph_max_pool(self, x):
        """
        Implementa il graph max pooling per ridurre la dimensionalità.
        
        Il pooling riduce il numero di geni di un fattore pool_size,
        mantenendo solo i valori massimi in finestre di dimensione pool_size.
        Questo aiuta a ridurre la complessità computazionale e catturare
        le feature più importanti.
        
        Args:
            x (torch.Tensor): Features input con shape [batch_size, num_genes, flm_gcn_dim_2]
        
        Returns:
            torch.Tensor: Features pooled con shape [batch_size, num_genes/pool_size, flm_gcn_dim_2]
        """
        
        if self.pool_size > 1:
            # Riorganizza per applicare max pooling 1D
            # [batch_size, num_genes, flm_gcn_dim_2] → [batch_size, flm_gcn_dim_2, num_genes]
            x = x.permute(0, 2, 1).contiguous()
            
            # Applica max pooling 1D lungo la dimensione dei geni
            # Riduce num_genes di un fattore pool_size
            x = nn.MaxPool1d(self.pool_size)(x)  # [batch_size, flm_gcn_dim_2, num_genes/pool_size]
            
            # Ripristina l'ordine delle dimensioni
            # [batch_size, flm_gcn_dim_2, num_genes/pool_size] → [batch_size, num_genes/pool_size, flm_gcn_dim_2]
            x = x.permute(0, 2, 1).contiguous()
            
            return x
        else:
            # Se pool_size = 1, non applica pooling
            return x

    def forward(self, feat, adj):
        """
        Forward pass del Feature Learning Module.
        
        Elabora le feature attraverso la pipeline completa:
        1. Multi-layer GCN per catturare relazioni tra geni
        2. Graph max pooling per riduzione dimensionalità
        3. Flatten layer per rappresentazione finale
        
        Args:
            feat (torch.Tensor): Matrice delle feature con shape [batch_size, num_genes, num_omics]
            adj (torch.sparse.FloatTensor): Matrice di adiacenza (knowledge graph biologico)
                                            con shape [num_genes, num_genes]
        
        Returns:
            torch.Tensor: Rappresentazione elaborata con shape [batch_size, flm_fl_dim]
        """
        
        # === FASE 1: MULTI-LAYER GCN ===
        # Applica convoluzione su grafo per catturare relazioni strutturali tra geni
        x = self.graph_conv_net(feat, adj)  # [batch_size, num_genes, flm_gcn_dim_2]

        # === FASE 2: GRAPH MAX POOLING ===
        # Riduce la dimensionalità mantenendo le feature più importanti
        x = self.graph_max_pool(x)  # [batch_size, num_genes/pool_size, flm_gcn_dim_2]

        # === FASE 3: FLATTEN LAYER ===
        # Appiattisce per layer lineare finale
        x = x.view(-1, self.fl_dim_i)  # [batch_size, num_genes/pool_size * flm_gcn_dim_2]
        
        # Applica trasformazione lineare finale con attivazione ReLU
        x = self.fl(x)     # [batch_size, flm_fl_dim]
        x = F.relu(x)      # Attivazione finale
        
        return x
