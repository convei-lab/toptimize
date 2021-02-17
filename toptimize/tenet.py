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

print_dataset_stat(dataset, data)

print_label_relation(data)

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

seed = 10
run = 0
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

input('Model Loading'+str('='*40))
print('Model\n', model, '\nOptimizer\n', optimizer)

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

input("Start Training "+str(run)+str('='*40))
best_val_acc = test_acc = 0
for epoch in range(0, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    # print(log.format(epoch, train_acc, best_val_acc, test_acc))
print('Run', run, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)
input("Finished Training "+str(run)+str('='*40))


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

compare_topology(pred_A, data, cm_filename='main'+str(run))
plot_tsne(prev_x, data.y, 'tsne_'+str(run)+'.png')
# plot_topology(A, data, 'A_original.png')
# plot_topology(gold_A, data, 'A_gold.png')
# plot_topology(sorted_gold_A, data, 'A_sorted_gold.png')
# plot_topology(A, data, 'A_sorted_original.png', sorting=True)
# plot_topology(gold_A, data, 'A_sorted_gold2.png', sorting=True)
# plot_sorted_topology_with_gold_topology(A, gold_A, data, 'A_original_with_gold_'+str(run)+'.png', sorting=False)
plot_sorted_topology_with_gold_topology(pred_A, gold_A, data, 'A_sorted_original_with_gold_'+str(run)+'.png', sorting=True)




val_accs, test_accs = [], []
for run in range(1, 5 + 1):
    print('Squaring Adj','='*40)
    dense_adj = to_dense_adj(data.edge_index)[0]
    print('dense_adj', dense_adj, dense_adj.shape)
    squared_adj = torch.matmul(dense_adj, dense_adj)
    print('squared_adj', squared_adj, squared_adj.shape)
    squared_adj[squared_adj>1]=1
    print('squared_adj (norm)', squared_adj, squared_adj.shape)
    squared_adj.fill_diagonal_(1)
    compare_topology(squared_adj, data, cm_filename='main'+str(run))

    squared_adj.fill_diagonal_(0)
    print('squared_adj (no_loop)', squared_adj, squared_adj.shape)

    squared_edge_index, squared_edge_weight = dense_to_sparse(squared_adj)
    print('squared_edge_index', squared_edge_index, squared_edge_index.shape)

    all_adj = torch.ones_like(squared_adj)
    all_edge_index, all_edge_weight = dense_to_sparse(all_adj)

    del squared_adj
    input()

    class Tenet(torch.nn.Module):
        def __init__(self):
            super(Tenet, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                                normalize=not args.use_gdc)
            self.tenet1 = TENET()
            self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                                normalize=not args.use_gdc)
            self.tenet2 = TENET()

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = self.conv1(x, edge_index, edge_weight)
            edge_score_1, edge_label_1 = self.tenet1(x, edge_index, edge_weight)
            x = F.relu(x)
            # x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
            edge_score_2, edge_label_2 = self.tenet2(x, squared_edge_index, squared_edge_weight)
            return x, F.log_softmax(x, dim=1)

        def infer(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = self.conv1(x, edge_index, edge_weight)
            edge_score_1 = self.tenet1.get_edge_score(x, all_edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight)
            edge_score_2 = self.tenet2.get_edge_score(x, all_edge_index)
            return edge_score_1, edge_score_2


    input('Model Loading'+str('='*40))
    model = Tenet().to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0),
        dict(params=model.tenet1.parameters(), weight_decay=5e-4),
        dict(params=model.tenet2.parameters(), weight_decay=5e-4),
    ], lr=0.01)
    print('Model\n', model, '\nOptimizer\n', optimizer)
    print('Model Parameterers')
    for name, param in model.named_parameters():
        print(name, param, 'grad', param.requires_grad)

    def train():
        model.train()
        optimizer.zero_grad()
        x, logits = model()

        task_loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        link_loss = TENET.get_link_prediction_loss(model)
        dist_loss = F.kl_div(logits, prev_logits, reduction = 'none', log_target = True).mean()
        total_loss = task_loss +  link_loss + 10 * dist_loss

        # print('Task loss', task_loss)
        # print('Link loss', link_loss)
        # print('Distillation loss', dist_loss)
        # print('Total loss', total_loss, '\n')

        total_loss.backward()
        optimizer.step()

    @torch.no_grad()
    def distillation():
        model.eval()
        x, logits = model()

        edge_score_1, edge_score_2 = model.infer()
        edge_score = 1* edge_score_1 +10 * edge_score_2
        print('edge_score_1', edge_score_1, edge_score_1.shape)
        print('edge_score_2', edge_score_2, edge_score_2.shape)
        print('edge_score', edge_score, edge_score.shape)

        # Add
        ############## Thresholding ################
        # edge_mask = edge_score > 30
        # new_edge_index = all_edge_index[:,edge_mask]
        ################ Sorting ###################
        sorted_score, mask_indices = torch.sort(edge_score, descending=True)
        new_edge_index = all_edge_index[:,mask_indices[:200000]]
        ################## Add #####################
        print('new_edge_index', new_edge_index, new_edge_index.shape)
        prev_num_edges = data.num_edges
        data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=-1)
        print('data.edge_index', data.edge_index, data.edge_index.shape)
        print('data.num_edges', data.num_edges)
        print()

        # Renorm
        dense_adj = to_dense_adj(data.edge_index)[0]
        print('dense_adj', dense_adj, dense_adj.shape)
        dense_adj[dense_adj>1] = 1
        print('dense_adj (norm)', dense_adj, dense_adj.shape)
        ################## Drop ####################
        del_edge_index = all_edge_index[:,mask_indices[-10000:]]
        print('del_edge_index', del_edge_index, del_edge_index.shape)
        for i, j in del_edge_index.T:
            dense_adj[i][j] = 0
            dense_adj[j][i] = 0
        #############################################
        edge_index, edge_weight = dense_to_sparse(dense_adj)
        print('edge_index', edge_index, edge_index.shape)
        print('edge_weight', edge_weight, edge_weight.shape)
        data.edge_index = edge_index
        data.edge_weight = edge_weight


        # Stats
        print('prev_num_edges', prev_num_edges)
        print('data.num_edges', data.num_edges)
        print('edge len difference', data.num_edges-prev_num_edges)

        pred = logits.max(1)[1]
        Y = F.one_hot(pred).float()
        YYT = torch.matmul(Y, Y.T)
        return x, logits, YYT

    input("\nStart Training "+str(run)+str('='*40)) 
    best_val_acc = test_acc = 0
    for epoch in range(0, 101):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print('Run', run, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)
    input("Finished Training "+str(run)+str('='*40))

    val_accs.append(best_val_acc)
    test_accs.append(test_acc)

    prev_x, prev_logits, YYT = distillation()

    A_temp = to_dense_adj(data.edge_index)[0]
    # A_temp.fill_diagonal_(1)
    A_temp[A_temp>1] = 1
    print('A difference', torch.where(A != A_temp), len(torch.where(A!=A_temp)[0]), len(torch.where(A!=A_temp)[1]))
    A = A_temp

    compare_topology(A_temp, data, cm_filename='main'+str(run))
    plot_tsne(prev_x, data.y, 'tsne_'+str(run)+'.png')
    plot_sorted_topology_with_gold_topology(A_temp, gold_A, data, 'A_sorted_original_with_gold_'+str(run)+'.png', sorting=True)








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

    gcn = GCN().to(device)
    optimizer = torch.optim.Adam([
        dict(params=gcn.conv1.parameters(), weight_decay=5e-4),
        dict(params=gcn.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.
    input('Model Loading'+str('='*40))
    print('Model\n', gcn, '\nOptimizer\n', optimizer)

    def gcntrain():
        gcn.train()
        optimizer.zero_grad()
        x, logits = gcn()
        F.nll_loss(logits[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()
    
    @torch.no_grad()
    def gcntest():
        gcn.eval()
        (x, logits), accs = gcn(), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    input("Vanila GCN Training"+str(run)+str('='*40))
    best_val_acc = test_acc = 0
    for epoch in range(0, 201):
        gcntrain()
        train_acc, val_acc, tmp_test_acc = gcntest()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print('Run', run, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)
    input("Finished Vanila Training "+str(run)+str('='*40))


# Analytics
print('Analytics')

val_accs = np.array(val_accs)
mean = np.mean(val_accs)
std = np.std(val_accs)
print('Val. Acc.:', round(mean,4), '+/-', str(round(std,4)))

test_accs = np.array(test_accs)
mean = np.mean(test_accs)
std = np.std(test_accs)
print('Test. Acc.:', round(mean,4), '+/-', str(round(std,4)))

print('val_accs', val_accs)
print('test_accs', test_accs)