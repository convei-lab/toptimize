import os.path as osp

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GNNExplainer, GATConv
from typing import Tuple, List, Dict
dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 16)
        self.conv2 = GATConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model_range = 5
node_range = 500

edge_binary_dict = {}
edge_entropy_dict = {}

for model_idx in range(model_range):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    x, edge_index = data.x, data.edge_index

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        log_logits = model(x, edge_index)
        loss = F.nll_loss(log_logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    explainer = GNNExplainer(model, epochs=200)

    for node_idx in range(node_range):
        node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index)
        
        print('model_idx', model_idx, 'node_idx', node_idx)
        for i, e in enumerate(edge_mask):
            if e > 0:
                target = data.edge_index[0][i].item()
                source = data.edge_index[1][i].item()
                edge_key = (target, source)
                edge_weight = round(e.item(), 2)
                edge_binary = -1 # Not skewed to 0 or 1 (e.g. 0.3 <= edge_weight <= 0.7)
                if edge_weight > 0.7:
                    edge_binary = 1
                elif edge_weight < 0.3:
                    edge_binary = 0
                
                # Debug
                # print('edge weight for ', '('+str(source)+str(',')+str(target)+')',':', edge_weight, edge_binary)

                if edge_binary >= 0:
                    # add a binary to dictionary
                    if edge_key in edge_binary_dict.keys():
                        edge_binary_dict[edge_key].append(edge_binary)
                    else:
                        edge_binary_dict[edge_key] = []
                        edge_binary_dict[edge_key].append(edge_binary)
        # Debug
        # print(edge_binary_dict)
        # print()

from scipy.stats import entropy

edge_entropy_dict = {}

for edge_key, edge_binary_list in edge_binary_dict.items():
    n = len(edge_binary_list)
    np = edge_binary_list.count(0)
    p = np / n

    nq = n - np
    q = nq / n

    assert np + nq == n

    edge_entropy = entropy([p, q], base=2)
    edge_entropy_dict[edge_key] = edge_entropy

# Debug
# print('final representation', edge_entropy_dict)

cnt = [{} for _ in range(5)]

for edge_key, edge_entropy in edge_entropy_dict.items():
    if 0 <= edge_entropy < 0.2:
        cnt[0][edge_key] = edge_entropy
    elif 0.2 <= edge_entropy < 0.4:
        cnt[1][edge_key] = edge_entropy
    elif 0.4 <= edge_entropy < 0.6:
        cnt[2][edge_key] = edge_entropy
    elif 0.6 <= edge_entropy < 0.8:
        cnt[3][edge_key] = edge_entropy
    elif 0.8 <= edge_entropy <= 1.0:
        cnt[4][edge_key] = edge_entropy
    else:
        raise Exception("Edge entropy value should be in [0, 1]")

# Counts
print()
print('Counts per bin')
total = len(edge_binary_dict)
confirm = 0
for i in range(5):
    print('Bin', i, ':', len(cnt[i]), 'Ratio:', float(len(cnt[i])) / total)
    confirm += len(cnt[i])
print()
assert confirm == total

# Examples
print('5 Examples per bin')
for i in range(5):
    print('Bin', i)
    for j, (edge_key, edge_entropy) in enumerate(cnt[i].items()):
        if j == 5:
            break
        print('Edge:', edge_key, 'Entropy:', edge_entropy, 'Binary', edge_binary_dict[edge_key])
    print()

