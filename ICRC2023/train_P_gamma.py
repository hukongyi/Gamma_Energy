import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from tqdm.notebook import tqdm
from dgl.data.utils import save_graphs, load_graphs

device = "cuda" if torch.cuda.is_available() else "cpu"

class Gamma_Allsky_dgl(DGLDataset):
    def __init__(self, path):
        self.graph, self.labels = load_graphs(path)

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, index):
        return (self.graph[index], self.labels["isgamma"][index])
    
class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN_MaxPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            MaxPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer

    
class GIN_SumPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer
    
class GIN_AvgPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            AvgPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer

model_name = ["GIN_MaxPooling","GIN_SumPooling","GIN_AvgPooling"]
result = dict()

k = 10
train_path = f"/cxtmp/hky/ICRCdata/cosmic_gamma_train_dgl_knn_{k}.bin"
val_path = f"/cxtmp/hky/ICRCdata/cosmic_gamma_val_dgl_knn_{k}.bin"

MCdataset_train = Gamma_Allsky_dgl(train_path)
MCdataset_val = Gamma_Allsky_dgl(val_path)

train_dataloader = GraphDataLoader(
    MCdataset_train, batch_size=256, drop_last=False, num_workers=4, shuffle=True
)
val_dataloader = GraphDataLoader(
    MCdataset_val, batch_size=256, drop_last=False, num_workers=4, shuffle=False
)

test= next(iter(train_dataloader))
test[0].to(device)
for i, modeltype in enumerate([GIN_MaxPooling,GIN_SumPooling,GIN_AvgPooling]):
    val_loss_best = 1
    model = modeltype(4, 16, 2).to(device)
    maxtpoch = 80
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, maxtpoch)
    lossfunction = nn.CrossEntropyLoss()

    for epoch in range(maxtpoch):
        model.train()
        for batched_graph, labels in train_dataloader:
            if batched_graph.num_edges() > 2e8:
                continue
            batched_graph, labels = batched_graph.to(device), labels.to(
                device
            )
            pred = model(batched_graph, batched_graph.ndata["xdata"])
            loss = lossfunction(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#             print(loss)
        lr_scheduler.step()
        y_pred = list()
        y_orgin = list()
        model.eval()
        with torch.no_grad():
            for batched_graph, labels in val_dataloader:
                if batched_graph.num_edges() > 2e8:
                    continue
                batched_graph, labels = batched_graph.to(device), labels.to(device)
                pred = model(batched_graph, batched_graph.ndata["xdata"])
                y_pred.append(pred.cpu().numpy())
                y_orgin.append(labels.cpu().numpy())
        y_pred = np.concatenate(y_pred,axis=0)
        y_orgin = np.concatenate(y_orgin)
        val_loss =lossfunction(torch.from_numpy(y_pred),torch.from_numpy(y_orgin))
#             print(val_loss)
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            torch.save(
                model.state_dict(), f"/cxtmp/hky/ICRCdata/{model_name[i]}_{k}_Adam_P_gamma.pt"
            )
            result[f"{model_name[i]}_{k}"] = val_loss_best
            print(f"{epoch} {model_name[i]}_{k}", val_loss_best)
