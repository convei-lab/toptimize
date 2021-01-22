import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GCN4Conv  # noqa
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt

import random
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

print()
print(f'Dataset: {dataset}:')
print('===========================================================================================================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print()
print(data)
print('===========================================================================================================')

# Gather some statistics about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

gold_Y = F.one_hot(data.y).float()
gold_A = torch.matmul(gold_Y, gold_Y.T)

print()
print('Label Relation')
print('============================================================')
print(f'Gold Label Y: {gold_Y}')
print(f'Shape: {gold_Y.shape}')
print(f'Transpose of Y: {gold_Y.T}')
print(f'Shape: {gold_Y.T.shape}')
print(f'Gold A: {gold_A}')
print(f'Shape: {gold_A.shape}')

def compare_topology(pred_A, gold_A, cm_filename='confusion_matrix_display'):
    flat_pred_A = pred_A.detach().cpu().view(-1)
    flat_gold_A = gold_A.detach().cpu().view(-1)
    conf_mat = confusion_matrix(y_true=flat_gold_A, y_pred=flat_pred_A)
    print('conf_mat', conf_mat, conf_mat.shape)
    print('conf_mat.ravel()', conf_mat.ravel(), conf_mat.ravel().shape)
    tn, fp, fn, tp = conf_mat.ravel()
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    f1 = 2*(ppv*tpr)/(ppv+tpr)
    print()
    print('Confusion Matrix')
    print('============================================================')
    print(f'Flatten A: {flat_pred_A}')
    print(f'Shape: {flat_pred_A.shape}')
    print(f'Number of Positive Prediction: {flat_pred_A.sum()} ({flat_pred_A.sum().true_divide(len(flat_pred_A))})')
    print(f'Flatten Gold A: {flat_gold_A}')
    print(f'Shape: {flat_gold_A.shape}')
    print(f'Number of Positive Class: {flat_gold_A.sum()} ({flat_gold_A.sum().true_divide(len(flat_gold_A))})')
    print(f'Confusion matrix: {conf_mat}')
    print(f'Raveled Confusion Matrix: {conf_mat.ravel()}')
    print(f'True positive: {tp} # 1 -> 1')
    print(f'False positive: {fp} # 0 -> 1')
    print(f'True negative: {tn} # 0 -> 0')
    print(f'False negative: {fn} # 1 -> 0')
    print(f'Precision: {ppv} # TP/(TP+FP)')
    print(f'Negative predictive value: {round(npv,2)} # TN/(TN+FN)')
    print(f'Recall: {tpr} # TP/P')
    print(f'Selectivity: {round(tnr,2)} # TN/N')
    print(f'F1 score: {f1}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.savefig(cm_filename+'.png')


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
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
    # input()
print('Run', run, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)

print("Finished Training", run, '\n')
input()

def plot_tsne(tsne_x, tsne_y, fig_name, label_names=None):
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    
    if tsne_x.is_cuda:
        tsne_x = tsne_x.detach().cpu()
    if tsne_y.is_cuda:
        tsne_y = tsne_y.detach().cpu()
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(tsne_x.cpu())

    target_ids = range(len(tsne_y))
    if not label_names:
        label_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g'] 

    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'y', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, label_names):
        plt.scatter(X_2d[tsne_y == i, 0], X_2d[tsne_y == i, 1], c=c, s=3, label=label)
    plt.legend()
    plt.savefig(fig_name)

prev_final, YYT, final_x = final_and_yyt_for_supervision()

A = to_dense_adj(data.edge_index)[0]
A.fill_diagonal_(1)
print('A', A, A.shape)
# print('ok?', torch.all(A==1 or A==0))
# compare_topology(A, gold_A, cm_filename='main'+str(run))
# plot_tsne(final_x, data.y, 'tsne_0.png')
input()











val_accs, test_accs = [], []

for run in range(1, 8 + 1):
    # denser_edge_index = torch.nonzero(YYT == 1, as_tuple=False)
    # denser_edge_index = denser_edge_index.t().contiguous()

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCN4Conv(dataset.num_features, 16, cached=True,
                                normalize=not args.use_gdc)
            self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                                normalize=not args.use_gdc)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = F.relu(self.conv1(x, edge_index, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            final = self.conv2(x, edge_index, edge_weight)
            return final, F.log_softmax(final, dim=1)

    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

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
        print('Task loss', task_loss)

        link_loss = GCN4Conv.get_link_prediction_loss(model)
        print('Link loss', link_loss)

        # redundancy_loss = F.mse_loss(final, prev_final, reduction = 'mean')
        redundancy_loss = F.kl_div(logits, prev_final, reduction = 'none', log_target = True).mean()
        #redundancy_loss = torch.distributions.kl.kl_divergence(logits, prev_final).sum(-1)
        print('Redundancy loss', redundancy_loss)

        total_loss = 1 * task_loss +  1 * link_loss + 10 * redundancy_loss
        print('Total loss', total_loss)

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
        new_edge = model.conv1.cache["new_edge"]
        new_edge = new_edge[:,:]
        print('new_edge', new_edge, new_edge.shape)
        new_edge_temp1 = new_edge[0].unsqueeze(0)
        new_edge_temp2 = new_edge[1].unsqueeze(0)
        new_edge_homo = torch.cat((new_edge_temp2, new_edge_temp1), dim=0)
        print('new_edge_homo', new_edge_homo, new_edge_homo.shape)
        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
        print('data.num_edges', data.num_edges)
        data.edge_index = torch.cat([data.edge_index, new_edge_homo], dim=1)
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
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print('Run', run, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)
    input()

    val_accs.append(best_val_acc)
    test_accs.append(test_acc)

    print("Finished Training", run, '\n')

    prev_final, YYT, final_x = final_and_yyt_for_supervision()
    A = to_dense_adj(data.edge_index)[0]
    A[A>1] = 1
    # with open('newA_' + str(run) + '.pickle', 'wb') as f:
    #     pickle.dump(A, f)
    A.fill_diagonal_(1)
    print('A', A, A.shape)
    # print('ok?', torch.all(A==1 or A==0))
    # compare_topology(A, gold_A, cm_filename='main'+str(run))
    # plot_tsne(final_x, data.y, 'tsne_'+str(run)+'.png')
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

print('Test. Acc.:', mean, '+/-', str(std))

print('Vals Accs: ', val_accs)
print('Test Accs', test_accs)