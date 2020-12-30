from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os.path as osp
import pickle

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

A = to_dense_adj(data.edge_index)[0]
A = A.fill_diagonal_(1)

gold_Y = F.one_hot(data.y).float()
gold_A = torch.matmul(gold_Y, gold_Y.T)

def compare_topology(pred_A, gold_A, cm_filename='confusion_matrix_A_add_TP'):
    flat_pred_A = pred_A.detach().cpu().view(-1)
    flat_gold_A = gold_A.detach().cpu().view(-1)
    conf_mat = confusion_matrix(y_true=flat_gold_A, y_pred=flat_pred_A)
    tn, fp, fn, tp = conf_mat.ravel()
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    f1 = 2*(ppv*tpr)/(ppv+tpr)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.savefig(cm_filename+'.png')


cnt = 0
for i in range(2708):
    for j in range(2708):
        if gold_A[i][j] == 1 and A[i][j] == 0:
            A[i][j] = 1

compare_topology(A, gold_A)
with open('A_add_TP.pickle', 'wb') as f:
    pickle.dump(A, f)
