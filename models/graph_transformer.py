"""Graph Transformer Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math

class GraphTransformer(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_classes=5, gat_heads=8, transformer_layers=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(num_features, hidden_dim, heads=gat_heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim*gat_heads, hidden_dim, heads=1, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, transformer_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, x, edge_index, batch=None):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = x.unsqueeze(0) if batch is None else x.mean(0, keepdim=True).unsqueeze(0)
        x = self.transformer(x)
        return self.classifier(x.mean(1))
