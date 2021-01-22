from abc import ABCMeta
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, TOP  # noqa
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt

import random
import numpy as np
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

print_dataset_stat(dataset, data)
input()

print_label_relation(data)
input()

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

seed = 0
run = 0
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
model, data = Net().to(device), data.to(device)
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
def final_and_yyt_for_supervision():
    model.eval()
    final, logits = model()
    pred = logits.max(1)[1]
    Y = F.one_hot(pred).float()
    YYT = torch.matmul(Y, Y.T)
    return final, logits, YYT

input("Start Training "+str(run))
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

input('Confusion Matrix '+str(run))
prev_final, prev_logits, YYT = final_and_yyt_for_supervision()

pred_A = to_dense_adj(data.edge_index)[0]
pred_A.fill_diagonal_(1)
compare_topology(pred_A, data, cm_filename='main'+str(run))

input('\nPlot TSNE')
plot_tsne(prev_final, data.y, 'tsne_gold.png')

input('\nPlot Topology')
A = to_dense_adj(data.edge_index)[0]
A.fill_diagonal_(1)
gold_Y = F.one_hot(data.y).float()
gold_A = torch.matmul(gold_Y, gold_Y.T)
sorted_gold_Y, sorted_Y_indices = torch.sort(data.y, descending=False)
sorted_gold_Y = F.one_hot(sorted_gold_Y).float()
sorted_gold_A = torch.matmul(sorted_gold_Y, sorted_gold_Y.T)

plot_topology(A, data, 'A_original.png')
plot_topology(gold_A, data, 'A_gold.png')
plot_topology(sorted_gold_A, data, 'A_sorted_gold.png')
plot_topology(A, data, 'A_sorted_original.png', sorting=True)
plot_topology(gold_A, data, 'A_sorted_gold2.png', sorting=True)
plot_sorted_topology_with_gold_topology(A, gold_A, data, 'A_original_with_gold_'+str(run)+'.png', sorting=False)
plot_sorted_topology_with_gold_topology(A, gold_A, data, 'A_sorted_original_with_gold_'+str(run)+'.png', sorting=True)




val_accs, test_accs = [], []

for run in range(1, 5 + 1):

    input("\nStart Training "+str(run)) 
    print('===========================================================================================================')

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
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

    model = Net().to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)

    print()
    print('Model', model)
    print('Optimizer', optimizer)

    def train():
        model.train()
        optimizer.zero_grad()
        final, logits = model()

        task_loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])

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
    def final_and_yyt_for_supervision():
        model.eval()
        final, logits = model()
        pred = logits.max(1)[1]
        Y = F.one_hot(pred).float()
        YYT = torch.matmul(Y, Y.T)
        return final, logits, YYT
    
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
    print('Run (middle)', run, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)

    prev_model = model






    print('===========================================================================================================')
    input('Training after link prediciton')
    class Net(torch.nn.Module):
        def __init__(self, x, edge_index, edge_weight):
            super(Net, self).__init__()
            self.top1 = TOP()
            self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                                normalize=not args.use_gdc)
            self.top2 = TOP()
            self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                                normalize=not args.use_gdc)
            self.x = x
            self.edge_index = edge_index
            self.edge_weight = edge_weight

        def forward(self):
            x, edge_index, edge_weight = self.x, self.edge_index, self.edge_weight
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            edge_index, edge_weight = self.top1(x, edge_index, edge_weight)
            x = F.dropout(x, training=self.training)
            final = self.conv2(x, edge_index, edge_weight)
            edge_index, edge_weight =  self.top2(x, edge_index, edge_weight)
            return final, F.log_softmax(final, dim=1)

        def add_new_edge(self):
            print('Original edge index', self.edge_index, self.edge_index.shape)
            print('New edge to add (layer 1)', self.top1.cache['new_edge'], self.top1.cache['new_edge'].shape)
            print('New edge to add (layer 2)', self.top2.cache['new_edge'], self.top2.cache['new_edge'].shape)
            self.edge_index = torch.cat([self.edge_index, self.top1.cache['new_edge']], dim=-1)
            self.edge_index = torch.cat([self.edge_index, self.top2.cache['new_edge']], dim=-1)
            print('Updated edges', self.edge_index, self.edge_index.shape)
            self.edge_weight = None

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = Net(data.x, data.edge_index, data.edge_attr).to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.top1.parameters(), weight_decay=0),
        dict(params=model.top2.parameters(), weight_decay=0),
    ], lr=0.1)

    # Loading the parameter from the previous model
    model.conv1.load_state_dict(prev_model.conv1.state_dict())
    model.conv2.load_state_dict(prev_model.conv2.state_dict())

    # freezing W in GCN
    # model.conv1.weight.requires_grad = False
    # model.conv1.bias.requires_grad = False
    # model.conv2.weight.requires_grad = False
    # model.conv2.bias.requires_grad = False

    print()
    print('Model', model)
    print('Optimizer', optimizer)
    print('Model Parameterers')
    for name, param in model.named_parameters():
        print(name, param, 'grad', param.requires_grad)

    def train():
        model.train()
        optimizer.zero_grad()
        final, logits = model()

        task_loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
        print('Task loss', task_loss)

        link_loss = TOP.get_link_prediction_loss(model)
        print('Link loss', link_loss)

        # redundancy_loss = F.mse_loss(final, prev_final, reduction = 'mean')
        redundancy_loss = F.kl_div(logits, prev_logits, reduction = 'none', log_target = True).mean()
        print('Redundancy loss', redundancy_loss)

        total_loss = task_loss +  link_loss + 10 * redundancy_loss
        # total_loss = task_loss +  link_loss
        # total_loss = task_loss + 10 * redundancy_loss
        print('Total loss', total_loss, '\n')

        total_loss.backward()
        optimizer.step()

    print('===========================================================================================================')
    input('Continue training '+str(run))

    best_val_acc = test_acc = 0
    for epoch in range(200, 401):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print('Run', run, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)

    val_accs.append(best_val_acc)
    test_accs.append(test_acc)

    print("Finished Training", run)

    input('\nAdd new edge')
    model.add_new_edge()
    print('model.edge_index', model.edge_index, model.edge_index.shape)

    input('\nConfusion matrix ' + str(run))
    A_temp = to_dense_adj(model.edge_index)[0]
    A_temp = A_temp.fill_diagonal_(1)
    A_temp[A_temp>1] = 1
    print('A difference', torch.where(A != A_temp), len(torch.where(A!=A_temp)[0]), len(torch.where(A!=A_temp)[1]))
    compare_topology(A_temp, data, cm_filename='main'+str(run))

    input('\nPlot TSNE')
    prev_final, prev_logits, YYT = final_and_yyt_for_supervision()
    plot_tsne(prev_final, data.y, 'tsne_gold.png')

    input('\nPlot topology')
    plot_sorted_topology_with_gold_topology(A_temp, gold_A, data, 'A_sorted_original_with_gold_'+str(run)+'.png', sorting=True)
    A = A_temp

    data.edge_index = model.edge_index


# Analytics
print('Analytics')

val_accs = np.array(val_accs)
mean = np.mean(val_accs)
std = np.std(val_accs)

print('Val. Acc.:', mean, '+/-', str(std))

test_accs = np.array(test_accs)
mean = np.mean(test_accs)
std = np.std(test_accs)

print('Test. Acc.:', mean, '+/-', str(std))
