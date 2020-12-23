import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GCN3Conv, ChebConv  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

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
        self.conv1 = GCN3Conv(dataset.num_features, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x, e_new = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1), e_new


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    logits, e_new = model()
    task_loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])
    link_pred_loss = F.binary_cross_entropy(e_new, torch.ones_like(e_new))
    loss = task_loss + 10 * link_pred_loss
    loss.backward()
    optimizer.step()
    # print(e_new)
    # input()


@torch.no_grad()
def test():
    model.eval()
    (logits, e_new), accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

import numpy as np
val_accs, test_accs = [], []
for i, run in enumerate(range(20)):
    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        # print(log.format(epoch, train_acc, best_val_acc, test_acc))
    print('Run', i, 'Val. Acc.', best_val_acc, 'Test Acc.', test_acc)

    val_accs.append(best_val_acc)
    test_accs.append(test_acc)

print('Done\n')

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