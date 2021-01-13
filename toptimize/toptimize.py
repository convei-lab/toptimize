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
print(f'Edge index: {data.edge_index} {data.edge_index.shape}')

A = to_dense_adj(data.edge_index)[0]
A.fill_diagonal_(1)
gold_Y = F.one_hot(data.y).float()
gold_A = torch.matmul(gold_Y, gold_Y.T)

sorted_gold_Y, sorted_Y_indices = torch.sort(data.y, descending=False)
sorted_gold_Y = F.one_hot(sorted_gold_Y).float()
sorted_gold_A = torch.matmul(sorted_gold_Y, sorted_gold_Y.T)

print()
print('Label Relation')
print('============================================================')
print(f'Original A: {A}')
print(f'Shape: {A.shape}')
print(f'Gold Label Y: {gold_Y}')
print(f'Shape: {gold_Y.shape}')
print(f'Transpose of Y: {gold_Y.T}')
print(f'Shape: {gold_Y.T.shape}')
print(f'Gold A: {gold_A}')
print(f'Shape: {gold_A.shape}')
print(f'Sorted gold Y: {sorted_gold_Y}')
print(f'Shape: {sorted_gold_Y.shape}')
print(f'Sorted gold Y indices: {sorted_Y_indices}')
print(f'Shape: {sorted_Y_indices.shape}')
print(f'Gold A: {sorted_gold_A}')
print(f'Shape: {sorted_gold_A.shape}')



def compare_topology(pred_A, gold_A, cm_filename='confusion_matrix_display'):
    flat_pred_A = pred_A.detach().cpu().view(-1)
    flat_gold_A = gold_A.detach().cpu().view(-1)
    conf_mat = confusion_matrix(y_true=flat_gold_A, y_pred=flat_pred_A)
    # print('conf_mat', conf_mat, conf_mat.shape)
    # print('conf_mat.ravel()', conf_mat.ravel(), conf_mat.ravel().shape)
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
    print(f'Precision: {round(ppv,5)} # TP/(TP+FP)')
    print(f'Negative predictive value: {round(npv,5)} # TN/(TN+FN)')
    print(f'Recall: {round(tpr,5)} # TP/P')
    print(f'Selectivity: {round(tnr,5)} # TN/N')
    print(f'F1 score: {f1}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.savefig(cm_filename+'.png')
    plt.clf()
    plt.cla()
    plt.close()


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
    plt.clf()
    plt.cla()
    plt.close()


def cpu(variable):
    if hasattr(variable, 'is_cuda') and variable.is_cuda:
        variable = variable.detach().cpu()
    return variable

def numpy(variable):
    if not isinstance(variable, np.ndarray):
        variable = variable.numpy()
    return variable

def sort_topology(adj):
    adj = adj[sorted_Y_indices] # sort row
    adj = adj[:, sorted_Y_indices] # sort column
    return adj

def plot_topology(adj, figname, sorting=False):
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(3000.0/float(DPI),3000.0/float(DPI))

    adj = cpu(adj)
    adj = numpy(adj)
    if sorting:
        adj = sort_topology(adj)

    plt.imshow(adj, cmap='hot', interpolation='none')    

    plt.savefig(figname)
    plt.clf()
    plt.cla()
    plt.close()

def zero_to_nan(adj):
    nan_adj = np.copy(adj).astype('float')
    nan_adj[nan_adj==0] = np.nan # Replace every 0 with 'nan'
    return nan_adj

def plot_sorted_topology_with_gold_topology(adj, gold_adj, figname, sorting=False):
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(3000.0/float(DPI),3000.0/float(DPI))

    adj = cpu(adj)
    adj = numpy(adj)
    gold_adj = cpu(gold_adj)
    gold_adj = numpy(gold_adj)

    if sorting:
        adj = sort_topology(adj)
        gold_adj = sort_topology(gold_adj)

    plt.imshow(gold_adj, cmap='hot', interpolation='none')

    intra_edge_mask = gold_adj == 1
    intra_edges = adj * intra_edge_mask
    # print('intra_edges', intra_edges, intra_edges.shape, np.sum(intra_edges))

    intra_edges = zero_to_nan(intra_edges)
    plt.imshow(intra_edges, cmap='bwr', interpolation='none')
    
    inter_edge_mask = ~intra_edge_mask
    inter_edges = adj * inter_edge_mask
    # print('inter_edges', inter_edges, inter_edges.shape, np.sum(inter_edges))

    inter_edges = zero_to_nan(inter_edges)
    plt.imshow(inter_edges, cmap='hsv', interpolation='none')
    
    plt.savefig(figname)
    plt.clf()
    plt.cla()
    plt.close()

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

A = to_dense_adj(data.edge_index)[0]
A.fill_diagonal_(1)
compare_topology(A, gold_A, cm_filename='main'+str(run))

input('\nPlot TSNE')
plot_tsne(prev_final, data.y, 'tsne_gold.png')

input('\nPlot Topology')
plot_topology(A, 'A_original.png')
plot_topology(gold_A, 'A_gold.png')
plot_topology(sorted_gold_A, 'A_sorted_gold.png')
plot_topology(A, 'A_sorted_original.png', sorting=True)
plot_topology(gold_A, 'A_sorted_gold2.png', sorting=True)
plot_sorted_topology_with_gold_topology(A, gold_A, 'A_original_with_gold_'+str(run)+'.png', sorting=False)
plot_sorted_topology_with_gold_topology(A, gold_A, 'A_sorted_original_with_gold_'+str(run)+'.png', sorting=True)








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
            edge_index, edge_weight = self.top1(x, edge_index, edge_weight)
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            edge_index, edge_weight =  self.top2(x, edge_index, edge_weight)
            x = F.dropout(x, training=self.training)
            final = self.conv2(x, edge_index, edge_weight)
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

    # activatie top1, top2
    model.top1.after_link_prediction = True
    model.top2.after_link_prediction = True

    print()
    print('Model', model)
    print('Optimizer', optimizer)
    print('Model Parameterers')
    for name, param in model.named_parameters():
        print(name, param, 'grad', param.requires_grad)
    input()


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
    compare_topology(A_temp, gold_A, cm_filename='main'+str(run))

    input('\nPlot TSNE')
    prev_final, prev_logits, YYT = final_and_yyt_for_supervision()
    plot_tsne(prev_final, data.y, 'tsne_gold.png')

    input('\nPlot topology')
    plot_sorted_topology_with_gold_topology(A_temp, gold_A, 'A_sorted_original_with_gold_'+str(run)+'.png', sorting=True)
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
