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
import pickle
import numpy as np

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
        return x, F.log_softmax(x, dim=1)

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

# A = to_dense_adj(data.edge_index)[0]

# edge_index0 = torch.nonzero(A == 1, as_tuple=False)
# A = A.fill_diagonal_(1)

# gold_Y = F.one_hot(data.y).float()
# A_add_TP = torch.matmul(gold_Y, gold_Y.T)
# print(A_add_TP)
###################################################################

# A = A.fill_diagonal_(0) # remove self-loop
# edge_index = torch.nonzero(gold_A == 1, as_tuple=False)


# data.edge_index = edge_index.t().contiguous()

with open('newA_300.pickle', 'rb') as f:
     A_add_TP = pickle.load(f)

# # A_add_TP.fill_diagonal_(1)

# # compare_topology(A_add_TP, gold_A, cm_filename='newA_1')

edge_index = torch.nonzero(A_add_TP == 1, as_tuple=False)
data.edge_index = edge_index.t().contiguous()


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, data = Net().to(device), data.to(device)
# optimizer = torch.optim.Adam([
#     dict(params=model.conv1.parameters(), weight_decay=5e-4),
#     dict(params=model.conv2.parameters(), weight_decay=0)
# ], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    a,b = model()
    pred, target = b[data.train_mask], data.y[data.train_mask]
    loss = F.nll_loss(pred, target)
    loss.backward()
    optimizer.step()
    # with open('gold_final_x.pickle', 'wb') as f:
    #     pickle.dump(a, f)
    return a, b


@torch.no_grad()
def test():
    model.eval()
    a, logits = model()
    accs = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = test_acc = 0
val_accs, test_accs = [], []
for i, run in enumerate(range(20)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.
    best_val_acc = test_acc = 0
    for epoch in range(1, 200):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, val_acc, tmp_test_acc))
    print(log.format(epoch, train_acc, best_val_acc, test_acc))

    val_accs.append(best_val_acc)
    test_accs.append(test_acc)

print('Done\n')

# Analytics
print('Task Loss + MSE Loss')

val_accs = np.array(val_accs)
mean = np.mean(val_accs)
std = np.std(val_accs)

print('Val. Acc.:', mean, '+/-', str(std))
test_accs = np.array(test_accs)
mean = np.mean(test_accs)
std = np.std(test_accs)

print('Test. Acc.:', mean, '+/-', str(std))



