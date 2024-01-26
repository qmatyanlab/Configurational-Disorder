import torch
import torch_geometric

import torch.nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import CGConv, GATv2Conv, TopKPooling
from torch_geometric.nn import global_mean_pool as gmp

class MLPembedding(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_layers):
        super().__init__()
        prevdim, hiddendim = input_dim, output_dim

        self.modules = []
        for _ in range(num_layers):
            self.modules.append(torch.nn.Linear(prevdim, hiddendim, bias=False))  # .cuda()
            prevdim = hiddendim

    def forward(self, x):
        if x.dim() == 1:
            x = x[:, None]
        for linear in self.modules[:-1]:
            x = F.relu(linear(x))
        x = self.modules[-1](x)
        return x

class CGNN(torch.nn.Module):
    def __init__(self, num_embedding_layers, hidden_channels, num_hidden_layers, num_node_features_input, num_edge_features, device=torch.device('cpu')):
        super().__init__()
        self.embedding = MLPembedding(num_node_features_input, hidden_channels, num_embedding_layers)
        self.Conv_list = [CGConv(channels=hidden_channels, dim=num_edge_features, batch_norm=True).to(device) for _ in range(num_hidden_layers)]

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.bn = BatchNorm1d(hidden_channels)
        self.dropout = Dropout(p=0.5)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, data):
        x = self.embedding(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        global_feature = 0
        for conv in self.Conv_list:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            global_feature += gmp(x, batch)

        global_feature = F.relu(self.bn(self.lin1(global_feature)))
        # x = self.dropout(x)
        global_feature = self.lin2(global_feature)
        return global_feature.squeeze()

class GAT(torch.nn.Module):
    def __init__(self, num_embedding_layers, hidden_channels, num_hidden_layers, num_node_features_input, num_edge_features, num_heads, device=torch.device('cpu')):
        super().__init__()
        self.embedding = MLPembedding(num_node_features_input, hidden_channels, num_embedding_layers)
        self.Conv_list = [
            GATv2Conv(hidden_channels, int(hidden_channels / num_heads), heads=num_heads, edge_dim=num_edge_features).to(device) for _
            in range(num_hidden_layers)]

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.bn = BatchNorm1d(hidden_channels)
        self.dropout = Dropout(p=0.5)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, data):
        x = self.embedding(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        global_feature = 0
        for conv in self.Conv_list:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            global_feature += gmp(x, batch)

        global_feature = F.relu(self.bn(self.lin1(global_feature)))
        # x = self.dropout(x)
        global_feature = self.lin2(global_feature)
        return global_feature.squeeze()