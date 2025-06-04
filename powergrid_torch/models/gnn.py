import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.nn import GraphConv, GCNConv, SAGEConv, GATConv, ChebConv
from powergrid_torch.models.baseClass import PowerGridModelBase


class GNNModel(PowerGridModelBase):
    def __init__(self, in_channels=5, hidden_channels=8, out_channels=2, n_bus=14, gnn_type='GCN', is_batch_norm='False', is_dropout=0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_bus = n_bus

        if gnn_type == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv_out = GCNConv(hidden_channels, out_channels)
        
        elif gnn_type == 'GraphConv':
            self.conv1 = GraphConv(in_channels, hidden_channels)
            self.conv2 = GraphConv(hidden_channels, hidden_channels)
            self.conv_out = GraphConv(hidden_channels, out_channels)

        elif gnn_type == 'SAGEConv':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv_out = SAGEConv(hidden_channels, out_channels)

        elif gnn_type == 'GATConv':
            self.conv1 = GATConv(in_channels, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv_out = GATConv(hidden_channels, out_channels)

        elif gnn_type == 'ChebConv':
            self.conv1 = ChebConv(in_channels, hidden_channels, K=2)
            self.conv2 = ChebConv(hidden_channels, hidden_channels, K=2)
            self.conv_out = ChebConv(hidden_channels, out_channels, K=2)

        else:
            raise ValueError("Choose between GCN, GraphConv, SAGEConv, GATConv, ChebConv")

        if is_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_channels)
            self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.linear = nn.Linear(hidden_channels*n_bus, out_channels*n_bus)
    
    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        
        out = self.conv1(x, edge_index)
        out = self.relu1(out)

        out = self.conv2(out, edge_index)
        out = self.relu2(out)

        batch_size = batch.num_graphs
        out = out.view(batch_size, self.n_bus*self.hidden_channels)
        out = self.linear(out)

        return out.view(batch_size*self.n_bus, 2)

