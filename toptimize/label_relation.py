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
A = A.fill_diagonal_(1)

print()
print('Citation Relation')
print('============================================================')
print(f'A: {A}')
print(f'Shape: {A.shape}')


Y = F.one_hot(data.y)
A_hat = torch.matmul(Y, Y.T)

print()
print('Label Relation')
print('============================================================')
print(f'Gold Label Y: {Y}')
print(f'Shape: {Y.shape}')
print(f'Transpose of Y: {Y.T}')
print(f'Shape: {Y.T.shape}')
print(f'Gold A: {A_hat}')
print(f'Shape: {A_hat.shape}')

flat_A = A.view(-1).long()
flat_A_hat = A_hat.view(-1)
conf_mat = confusion_matrix(y_true=flat_A_hat, y_pred=flat_A)
tn, fp, fn, tp = conf_mat.ravel()
ppv = tp/(tp+fp)
npv = tn/(tn+fn)
tpr = tp/(tp+fn)
tnr = tn/(tn+fp)
f1 = 2*(ppv*tpr)/(ppv+tpr)
print()
print('Confusion Matrix')
print('============================================================')
print(f'Flatten A: {flat_A}')
print(f'Shape: {flat_A.shape}')
print(f'Number of Positive Prediction: {flat_A.sum()} ({flat_A.sum().true_divide(len(flat_A))})')
print(f'Flatten Gold A: {flat_A_hat}')
print(f'Shape: {flat_A_hat.shape}')
print(f'Number of Positive Class: {flat_A_hat.sum()} ({flat_A_hat.sum().true_divide(len(flat_A_hat))})')
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
plt.savefig('confusion_matrix_display.png')

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
def infer():
    model.eval()
    logits = model()
    pred = logits.max(1)[1]
    Y_prime = F.one_hot(pred).float()
    A_prime = torch.matmul(Y_prime, Y_prime.T)
    return Y_prime, A_prime

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

Y_prime, A_prime = infer()
print()
print('Infer')
print('============================================================')
print(f'Y_prime: {Y_prime} {Y_prime.shape}')
print(f'A_prime: {A_prime} {A_prime.shape}')

flat_A = A_prime.view(-1).cpu().detach().long()
conf_mat = confusion_matrix(y_true=flat_A_hat, y_pred=flat_A)
tn, fp, fn, tp = conf_mat.ravel()
ppv = tp/(tp+fp)
npv = tn/(tn+fn)
tpr = tp/(tp+fn)
tnr = tn/(tn+fp)
f1 = 2*(ppv*tpr)/(ppv+tpr)
print()
print('Confusion Matrix')
print('============================================================')
print(f'Flatten A: {flat_A}')
print(f'Shape: {flat_A.shape}')
print(f'Number of Positive Prediction: {flat_A.sum()} ({flat_A.sum().true_divide(len(flat_A))})')
print(f'Flatten Gold A: {flat_A_hat}')
print(f'Shape: {flat_A_hat.shape}')
print(f'Number of Positive Class: {flat_A_hat.sum()} ({flat_A_hat.sum().true_divide(len(flat_A_hat))})')
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
