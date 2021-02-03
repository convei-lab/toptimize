from abc import ABCMeta
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, TENET, GCN4Conv, dense  # noqa
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.sparse import dense_to_sparse
import matplotlib.pyplot as plt

import random
import numpy as np
from utils import print_dataset_stat, print_label_relation, compare_topology, plot_tsne, plot_topology, plot_sorted_topology_with_gold_topology

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

# print_dataset_stat(dataset, data)

# print_label_relation(data)

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

seed = 0
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        final = self.conv2(x, edge_index, edge_weight)
        return final, F.log_softmax(final, dim=1)
        
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = GCN().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.

def train():
    model.train()
    optimizer.zero_grad()
    x, logits = model()
    F.nll_loss(logits[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

@torch.no_grad()
def test():
    model.eval()
    (x, logits), accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

@torch.no_grad()
def distillation():
    model.eval()
    x, logits = model()
    pred = logits.max(1)[1]
    Y = F.one_hot(pred).float()
    YYT = torch.matmul(Y, Y.T)
    return x, logits, YYT

def training(edge_index):
    global model, data
    data.edge_index = edge_index

    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # print(log.format(epoch, train_acc, best_val_acc, test_acc))
    return test_acc

print(training(data.edge_index))
prev_x, prev_logits, YYT = distillation()
pred_A = to_dense_adj(data.edge_index)[0]
pred_A.fill_diagonal_(1)

A = to_dense_adj(data.edge_index)[0]
A.fill_diagonal_(1)
gold_Y = F.one_hot(data.y).float()
gold_A = torch.matmul(gold_Y, gold_Y.T)
sorted_gold_Y, sorted_Y_indices = torch.sort(data.y, descending=False)
sorted_gold_Y = F.one_hot(sorted_gold_Y).float()
sorted_gold_A = torch.matmul(sorted_gold_Y, sorted_gold_Y.T)

# compare_topology(pred_A, data, cm_filename='main')
# plot_tsne(prev_x, data.y, 'tsne_'+str(run)+'.png')
# plot_topology(A, data, 'A_original.png')
# plot_topology(gold_A, data, 'A_gold.png')
# plot_topology(sorted_gold_A, data, 'A_sorted_gold.png')
# plot_topology(A, data, 'A_sorted_original.png', sorting=True)
# plot_topology(gold_A, data, 'A_sorted_gold2.png', sorting=True)
# plot_sorted_topology_with_gold_topology(A, gold_A, data, 'A_original_with_gold_'+str(run)+'.png', sorting=False)
# plot_sorted_topology_with_gold_topology(pred_A, gold_A, data, 'A_sorted_original_with_gold_'+str(run)+'.png', sorting=True)

def get_indices_for(A_gold, A):
    A_black = torch.ones_like(A_gold) - A_gold
    A_neg = torch.ones_like(A) - A
    TP = torch.nonzero(A_gold * A > 0, as_tuple=False).T
    FP = torch.nonzero(A_black * A > 0, as_tuple=False).T
    TN = torch.nonzero(A_black * A_neg > 0, as_tuple=False).T
    FN = torch.nonzero(A_gold * A_neg > 0, as_tuple=False).T
    return TP, FP, TN, FN

def toggle_topology_randomly(A, target_indices, prob):
    num_samples = target_indices.size(1) * prob
    print('num_samples', num_samples, num_samples.shape)
    choice = torch.multinomial(target_indices.float(), num_samples)
    print('choice', choice, choice.shape)
    for i, j in target_indices[:,choice].T:
        print(i, j)
        break
        if A[i][j] == 1:
            A[i][j] = 0
        else:
            A[i][j] = 1
    return A

TP, FP, TN, FN = get_indices_for(gold_A, A)
print('TP', TP, TP.shape)
print('FP', FP, FP.shape)
print('TN', TN, TN.shape)
print('FN', FN, FN.shape)

toggle_topology_randomly(A, FP, 1)
toggle_topology_randomly(A, FN, 1)

