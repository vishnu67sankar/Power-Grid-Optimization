import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from powergrid_torch.models.baseClass import PowerGridModelBase
from torch_geometric.nn import TransformerConv 

class GraphTransformerModel(PowerGridModelBase):
    
    def __init__(self, in_channels, hidden_channels, out_channels, n_transformer_layers=1, n_heads=3, concat_heads=True, dropout_rate=.0, n_bus=14):

        super().__init__()
        self.n_bus = n_bus
        self.out_channels = out_channels

        self.transformer_layers = nn.ModuleList()
        current_channels = in_channels

        for i in range(n_transformer_layers):
            if concat_heads:
                transformer_out_channels = hidden_channels*n_heads
            else:
                transformer_out_channels = hidden_channels
            
            conv = TransformerConv(in_channels=current_channels, out_channels=hidden_channels, heads=n_heads, concat=concat_heads, dropout=dropout_rate)

            self.transformer_layers.append(conv)
            current_channels = transformer_out_channels

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.final_projection = nn.Linear(current_channels, out_channels)

    def forward(self, batch):
        """
        Forward pass for the PowerGridGraphTransformerModel.

        Args:
            batch: PyTorch Geometric Batch object.
                   batch.x: Node features, shape (total_nodes_in_batch, in_channels)
                   batch.edge_index: Edge connectivity

        Returns:
            Tensor: Final predictions.
                    Shape: (total_nodes_in_batch, out_channels)
        """

        x, edge_index = batch.x, batch.edge_index
        
        for i, conv_layer in enumerate(self.transformer_layers):
            x = conv_layer(x, edge_index)
            if i < len(self.transformer_layers) - 1: # if n_transformer_layers = 1, x is out of that layer.
                x = self.relu(x)
                x = self.dropout(x)

        if len(self.transformer_layers) > 0 : # if n_transformer_layers > 1, x is out of last layer (with activation from previous).
             x = self.relu(x)
             x = self.dropout(x)

        out = self.final_projection(x)
        return out
