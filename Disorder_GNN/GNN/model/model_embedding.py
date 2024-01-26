import torch
import torch_geometric

import torch.nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import CGConv, GATv2Conv, TransformerConv, NNConv
from torch_geometric.nn import global_mean_pool as gmp

class CGNN(torch.nn.Module):
    def __init__(self, num_hidden_layers, num_hidden_channels, num_edge_features, device=torch.device('cpu')):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=100, embedding_dim=num_hidden_channels).to(device)
        self.Conv_list = [CGConv(channels=num_hidden_channels, dim=num_edge_features, batch_norm=True).to(device) for _ in range(num_hidden_layers)]

        self.lin1 = Linear(num_hidden_channels, num_hidden_channels)
        self.bn = BatchNorm1d(num_hidden_channels)
        self.dropout = Dropout(p=0.5)
        self.lin2 = Linear(num_hidden_channels, 1)

    def forward(self, data):
        x = self.embedding(data.x.long())
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        global_feature = 0
        for conv in self.Conv_list:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            global_feature += gmp(x, batch)

        global_feature = F.relu(self.lin1(global_feature))
        # x = self.dropout(x)
        global_feature = self.lin2(global_feature)

        return global_feature.squeeze()

class GAT(torch.nn.Module):
    def __init__(self, num_hidden_layers, num_hidden_channels, num_heads, num_edge_features, device=torch.device('cpu')):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=100, embedding_dim=num_hidden_channels).to(device)
        self.Conv_list = [
            GATv2Conv(num_hidden_channels, int(num_hidden_channels / num_heads), heads=num_heads, edge_dim=num_edge_features).to(device) for _
            in range(num_hidden_layers)]

        self.lin1 = Linear(num_hidden_channels, num_hidden_channels)
        self.bn = BatchNorm1d(num_hidden_channels)
        self.dropout = Dropout(p=0.5)
        self.lin2 = Linear(num_hidden_channels, 1)

    def forward(self, data):
        x = self.embedding(data.x.long())
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        global_feature = 0
        for conv in self.Conv_list:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            global_feature += gmp(x, batch)

        global_feature = F.relu(self.lin1(global_feature))
        # x = self.dropout(x)
        global_feature = self.lin2(global_feature)
        return global_feature.squeeze()

class Transformer(torch.nn.Module):
    def __init__(self, num_hidden_layers, num_hidden_channels, num_heads, num_edge_features, device=torch.device('cpu')):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=100, embedding_dim=num_hidden_channels).to(device)
        self.Conv_list = [
            TransformerConv(num_hidden_channels, int(num_hidden_channels / num_heads), heads=num_heads, edge_dim=num_edge_features).to(device) for _
            in range(num_hidden_layers)]

        self.lin1 = Linear(num_hidden_channels, num_hidden_channels)
        self.bn = BatchNorm1d(num_hidden_channels)
        self.dropout = Dropout(p=0.5)
        self.lin2 = Linear(num_hidden_channels, 1)

    def forward(self, data):
        x = self.embedding(data.x.long())
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        global_feature = 0
        for conv in self.Conv_list:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            global_feature += gmp(x, batch)

        global_feature = F.relu(self.lin1(global_feature))
        # x = self.dropout(x)
        global_feature = self.lin2(global_feature)
        return global_feature.squeeze()

class MPNN(torch.nn.Module):
    def __init__(self, num_hidden_layers, num_hidden_channels, num_edge_features, device=torch.device('cpu')):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=100, embedding_dim=num_hidden_channels).to(device)
        nn = torch.nn.Sequential(torch.nn.Linear(num_edge_features, num_edge_features),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(num_edge_features, num_hidden_channels**2))
        self.Conv_list = [NNConv(in_channels=num_hidden_channels, out_channels=num_hidden_channels, nn=nn).to(device) for _ in range(num_hidden_layers)]

        self.lin1 = Linear(num_hidden_channels, num_hidden_channels)
        self.bn = BatchNorm1d(num_hidden_channels)
        self.dropout = Dropout(p=0.5)
        self.lin2 = Linear(num_hidden_channels, 1)

    def forward(self, data):
        x = self.embedding(data.x.long())
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        global_feature = 0
        for conv in self.Conv_list:
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            global_feature += gmp(x, batch)

        global_feature = F.relu(self.lin1(global_feature))
        # x = self.dropout(x)
        global_feature = self.lin2(global_feature)
        return global_feature.squeeze()
