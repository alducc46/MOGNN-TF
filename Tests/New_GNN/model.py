import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, ChebConv, global_mean_pool, global_max_pool
from torch.nn import BatchNorm1d, Dropout, Linear

# =============================================================================
# Test model
# =============================================================================

class SimpleGCN1(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(SimpleGCN1, self).__init__()
        # Layer di preprocessing (Come multimodal)
        self.pre_conv_linear_gene = torch.nn.Linear(num_node_features, 8)

        self.conv1 = ChebConv(8, hidden_dim, K=5)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)

        self.conv2 = ChebConv(hidden_dim, hidden_dim, K=5)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        self.dropout = Dropout(0.2)
        self.classifier = Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        # PREPROCESSING!
        x = self.pre_conv_linear_gene(x)  # (batch_size*1000, 2) -> (batch_size*1000, 8) Come in multimodal
        x = F.relu(x)
        # ---- Layer 1 ----
        x = self.conv1(x, edge_index, edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # ---- Layer 2 ----
        x = self.conv2(x, edge_index, edge_weight)
        x = self.bn2(x)
        x = F.relu(x)

        # ---- Pooling ----
        x = global_max_pool(x, batch)

        # ---- Classification ----
        return self.classifier(x)