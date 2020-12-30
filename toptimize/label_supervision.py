import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv  # noqa
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print('Create new A from Y*Y.T of the teacher model')
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()


dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data =  data.to(device)

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)

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
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

print()
print(f'Dataset: {dataset}:')
print('============================================================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print()
print(data)
print('============================================================')

print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Number of training nodes: {data.train_mask.sum()}')
print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print(f'Data dictionary: {data.__dict__}')

A = to_dense_adj(data.edge_index)[0]

edge_index0 = torch.nonzero(A == 1, as_tuple=False)
A = A.fill_diagonal_(1)

print()
print('Citation Relation')
print('============================================================')
print(f'A: {A}')
print(f'Shape: {A.shape}')


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
    print(f'Precision: {round(ppv,2)} # TP/(TP+FP)')
    print(f'Negative predictive value: {round(npv,2)} # TN/(TN+FN)')
    print(f'Recall: {round(tpr,2)} # TP/P')
    print(f'Selectivity: {round(tnr,2)} # TN/N')
    print(f'F1 score: {f1}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.savefig(cm_filename+'.png')

z = 0
compare_topology(A, gold_A, cm_filename='confusion_matrix_display'+str(z))



for z in range(1, 3 + 1):
    A = A.fill_diagonal_(0) # remove self-loop
    print()
    print('Running supervision', z)
    print(f'With A: {A}')
    print('=================================================================================================================')
    edge_index = torch.nonzero(gold_A == 1, as_tuple=False)
    # if z == 1:
    #     assert torch.all(torch.eq(edge_index0, edge_index))
    print(f'Edge index: {edge_index}')
    data.edge_index = edge_index.t().contiguous()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.

    print()
    print('Training')
    print('============================================================')
    print(f'Model: {model}')
    print(f'Optimizer: {optimizer}')

    def train():
        model.train()
        optimizer.zero_grad()
        pred, target = model()[data.train_mask], data.y[data.train_mask]
        F.nll_loss(pred, target).backward()
        optimizer.step()


    @torch.no_grad()
    def test():
        model.eval()
        logits, accs = model(), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    @torch.no_grad()
    def new_topology():
        model.eval()
        logits = model()
        print(f'Logits: {logits}')
        pred = logits.max(1)[1]
        new_Y = F.one_hot(pred).float()
        new_A = torch.matmul(new_Y, new_Y.T)
        return new_Y, new_A

    print()
    print('Logging')
    print('============================================================')
    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    new_Y, new_A = new_topology()
    print()
    print('New Topology')
    print('============================================================')
    print(f'new_Y: {new_Y} {new_Y.shape}')
    print(f'new_A: {new_A} {new_A.shape}')
    print(f'Previous A: {A} {A.shape}')
    print(f'Did any new changes?: {torch.any(A != new_A)}')
    changes = torch.nonzero(A != new_A, as_tuple=False)
    print(f'Changes: {changes} (of length {len(changes)})')

    compare_topology(new_A, gold_A, cm_filename='confusion_matrix_display'+str(z))

    A = new_A