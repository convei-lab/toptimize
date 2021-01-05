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

A = to_dense_adj(data.edge_index)[0]


edge_index0 = torch.nonzero(A == 1, as_tuple=False)
A = A.fill_diagonal_(1)

for z in range(1, 2 + 1):
    # A = A.fill_diagonal_(0) # remove self-loop
    edge_index = torch.nonzero(A == 1, as_tuple=False)
    # if z == 1:
    #     assert torch.all(torch.eq(edge_index0, edge_index))
    data.edge_index = edge_index.t().contiguous()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.

    def train():
        model.train()
        optimizer.zero_grad()
        final_x, logits = model()
        pred, target = logits[data.train_mask], data.y[data.train_mask]
        F.nll_loss(pred, target).backward()
        optimizer.step()

        return final_x, logits


    @torch.no_grad()
    def test():
        model.eval()
        final_x, logits = model()
        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    @torch.no_grad()
    def new_topology():
        model.eval()
        final_x, logits = model()
        print(f'Logits: {logits}')
        pred = logits.max(1)[1]
        new_Y = F.one_hot(pred).float()
        new_A = torch.matmul(new_Y, new_Y.T)
        return new_Y, new_A

    best_val_acc = test_acc = 0
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()

        if epoch == 200 and z == 2:
            print("@@@@@@@@@@@@@@@@@@@@@in@@@@@@@@@@@@@@@@@@")
            final_x, logits = train()
            with open('yyt_final_x.pickle', 'wb') as f:
                pickle.dump(final_x, f)

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

    with open('yyt_A.pickle', 'wb') as f:
        pickle.dump(new_A, f)
    A = new_A