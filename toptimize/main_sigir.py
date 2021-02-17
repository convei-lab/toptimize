import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GCN4ConvSIGIR, GATConv, GAT4ConvSIGIR  # noqa
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.sparse import dense_to_sparse
import matplotlib.pyplot as plt
from utils import print_dataset_stat, print_label_relation, compare_topology, plot_tsne, plot_topology, plot_sorted_topology_with_gold_topology

import random
import numpy as np
import pickle

import logging
from pathlib import Path
import sys
import atexit

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

method_name = 'GCNConv4SIGIR'
dataset_name = 'Cora'
basemodel = 'GCN'
path_name = method_name + '_' + dataset_name + '_' + basemodel

# Path
exp_path = (Path.cwd() / '..' / 'experiment' / path_name).resolve()
exp_path.mkdir(mode=0o777, parents=True, exist_ok=True)

# Log
log_path = exp_path / str(path_name+'.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG)
print_to_log_file = False
if print_to_log_file:
    log_file = open(log_path, "w")
    sys.stdout = log_file  # change default print() to write into log file

    def clear():
        sys.stdout.close()
    atexit.register(clear)

# Data
data_path = osp.join(osp.dirname(osp.realpath(__file__)),
                     '..', 'data', dataset_name)
dataset = Planetoid(data_path, dataset_name, transform=T.NormalizeFeatures())
data = dataset[0]

# Print Data Info.
print_dataset_stat(dataset, data)
print_label_relation(data)

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

# print('data.edge_index', data.edge_index, data.edge_index.shape)
# mask = torch.randint(0, 3 + 1, (1, data.edge_index.size(1)))[0]
# print('mask', mask, mask.shape)
# mask = mask >= 3
# print('mask', mask, mask.shape)
# data.edge_index = data.edge_index[:, mask]
# print('data.edge_index', data.edge_index, data.edge_index.shape)
# input()


run = 0
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


class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        final = self.conv2(x, data.edge_index)
        return final, F.log_softmax(final, dim=1)


random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
if basemodel == 'GCN':
    model = GCN().to(device)
elif basemodel == 'GAT':
    model = GAT().to(device)
data = data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.

print()
print('Model', model)
print('Optimizer', optimizer)


def train():
    model.train()
    optimizer.zero_grad()
    final, logits = model()

    task_loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    # print('Task loss', task_loss)

    total_loss = task_loss

    total_loss.backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    (final, logits), accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


@torch.no_grad()
def distillation():
    model.eval()
    final, logits = model()
    pred = logits.max(1)[1]
    Y = F.one_hot(pred).float()
    YYT = torch.matmul(Y, Y.T)
    return logits, YYT, final


print("Start Training", run)
print('===========================================================================================================')
best_val_acc = test_acc = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(epoch, train_acc, best_val_acc, test_acc))
    # input()
print('Run', run, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)

print("Finished Training", run, '\n')
input()

prev_final, YYT, final_x = distillation()

pred_A = to_dense_adj(data.edge_index)[0]
pred_A.fill_diagonal_(1)

A = to_dense_adj(data.edge_index)[0]
A.fill_diagonal_(1)
gold_Y = F.one_hot(data.y).float()
gold_A = torch.matmul(gold_Y, gold_Y.T)
# sorted_gold_Y, sorted_Y_indices = torch.sort(data.y, descending=False)
# sorted_gold_Y = F.one_hot(sorted_gold_Y).float()
# sorted_gold_A = torch.matmul(sorted_gold_Y, sorted_gold_Y.T)

# compare_topology(pred_A, data, cm_filename='main'+str(run))
# plot_tsne(final_x, data.y, 'tsne_'+str(run)+'.png')
# plot_topology(A, data, 'A_original.png')
# plot_topology(gold_A, data, 'A_gold.png')
# plot_topology(sorted_gold_A, data, 'A_sorted_gold.png')
# plot_topology(A, data, 'A_sorted_original.png', sorting=True)
# plot_topology(gold_A, data, 'A_sorted_gold2.png', sorting=True)
# plot_sorted_topology_with_gold_topology(A, gold_A, data, 'A_original_with_gold_'+str(run)+'.png', sorting=False)
# plot_sorted_topology_with_gold_topology(pred_A, gold_A, data, 'A_sorted_original_with_gold_'+str(run)+'.png', sorting=True)


val_accs, test_accs = [], []

for run in range(1, 20 + 1):
    # denser_edge_index = torch.nonzero(YYT == 1, as_tuple=False)
    # denser_edge_index = denser_edge_index.t().contiguous()

    class GCN(torch.nn.Module):
        def __init__(self):
            super(GCN, self).__init__()
            self.conv1 = GCN4ConvSIGIR(dataset.num_features, 16, cached=True,
                                       normalize=not args.use_gdc)
            self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                                 normalize=not args.use_gdc)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            final = self.conv2(x, edge_index, edge_weight)
            return final, F.log_softmax(final, dim=1)

    class GAT(torch.nn.Module):
        def __init__(self):
            super(GAT, self).__init__()
            self.conv1 = GAT4ConvSIGIR(
                dataset.num_features, 8, heads=8, dropout=0.6)
            # On the Pubmed dataset, use heads=8 in conv2.
            self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                                 dropout=0.6)

        def forward(self):
            x = F.dropout(data.x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, data.edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            final = self.conv2(x, data.edge_index)
            return final, F.log_softmax(final, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if basemodel == 'GCN':
        model = GCN().to(device)
    elif basemodel == 'GAT':
        model = GAT().to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.

    print('Model', model)
    print('Optimizer', optimizer)

    def train():
        model.train()
        optimizer.zero_grad()
        final, logits = model()

        task_loss = F.nll_loss(
            logits[data.train_mask], data.y[data.train_mask])
        # print('Task loss', task_loss)

        link_loss = 0
        if basemodel == 'GCN':
            link_loss = GCN4ConvSIGIR.get_link_prediction_loss(model)
        elif basemodel == 'GAT':
            link_loss = GAT4ConvSIGIR.get_link_prediction_loss(model)
        # print('Link loss', link_loss)

        # redundancy_loss = F.mse_loss(final, prev_final, reduction = 'mean')
        redundancy_loss = F.kl_div(
            logits, prev_final, reduction='none', log_target=True).mean()
        #redundancy_loss = torch.distributions.kl.kl_divergence(logits, prev_final).sum(-1)
        # print('Redundancy loss', redundancy_loss)

        lambda1 = 0 if method_name.endswith('woLL') else 1
        lambda2 = 0 if method_name.endswith('woDL') else 10

        total_loss = 1 * task_loss + lambda1 * link_loss + lambda2 * redundancy_loss
        # print('Total loss', total_loss)

        total_loss.backward()
        optimizer.step()

    @torch.no_grad()
    def test():
        model.eval()
        (final, logits), accs = model(), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    @torch.no_grad()
    def distillation():
        model.eval()
        final, logits = model()

        print('original data.num_edges', data.num_edges)

        new_edge = model.conv1.cache["new_edge"]
        new_edge = new_edge[:, :]
        print('new_edge', new_edge, new_edge.shape)
        new_edge_temp1 = new_edge[0].unsqueeze(0)
        new_edge_temp2 = new_edge[1].unsqueeze(0)
        new_edge_homo = torch.cat((new_edge_temp2, new_edge_temp1), dim=0)
        print('new_edge_homo', new_edge_homo, new_edge_homo.shape)
        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
        data.edge_index = torch.cat([data.edge_index, new_edge_homo], dim=1)
        print('data.num_edges', data.num_edges)

        # reweight
        dense_adj = to_dense_adj(data.edge_index)[0]
        print('dense_adj', dense_adj, dense_adj.shape)
        dense_adj[dense_adj > 1] = 1
        print('dense_adj (norm)', dense_adj, dense_adj.shape)

        # # drop
        # del_edge = model.conv1.cache["del_edge"]
        # print('del_edge', del_edge, del_edge.shape)
        # for i, j in del_edge.T:
        #     dense_adj[i][j] = 0
        #     dense_adj[j][i] = 0

        # sparse
        edge_index, edge_weight = dense_to_sparse(dense_adj)
        print('edge_index', edge_index, edge_index.shape)
        print('edge_weight', edge_weight, edge_weight.shape)
        data.edge_index = edge_index
        data.edge_weight = edge_weight

        print('data.num_edges', data.num_edges)

        pred = logits.max(1)[1]
        Y = F.one_hot(pred).float()
        YYT = torch.matmul(Y, Y.T)
        return logits, YYT, final

    print("Start Training", run)
    print('===========================================================================================================')
    best_val_acc = test_acc = 0
    for epoch in range(1, 301):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print('Run', run, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)
    input()

    val_accs.append(best_val_acc)
    test_accs.append(test_acc)

    print("Finished Training", run, '\n')

    prev_final, YYT, final_x = distillation()
    A = to_dense_adj(data.edge_index)[0]
    A[A > 1] = 1
    # with open('newA_' + str(run) + '.pickle', 'wb') as f:
    #     pickle.dump(A, f)
    A.fill_diagonal_(1)
    print('A', A, A.shape)

    # compare_topology(A, data, cm_filename='main'+str(run))
    # plot_tsne(final_x, data.y, 'tsne_'+str(run)+'.png')
    # plot_sorted_topology_with_gold_topology(A, gold_A, data, 'A_sorted_original_with_gold_'+str(run)+'.png', sorting=True)
    input()

# Analytics
print('Task Loss + Link Prediction Loss')
print('Dataset split is the public fixed split')

val_accs = np.array(val_accs)
mean = np.mean(val_accs)
std = np.std(val_accs)

print('Val. Acc.:', mean, '+/-', str(std))


test_accs = np.array(test_accs)
mean = np.mean(test_accs)
std = np.std(test_accs)

print('Test. Acc.:', round(mean, 3), '+/-', str(round(std, 3)))
print('Vals Accs: ', val_accs)
print('Test Accs', test_accs)
