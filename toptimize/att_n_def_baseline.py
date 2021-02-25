import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph.utils import *
from deeprobust.graph.data import Dataset
import argparse
from torch_geometric.utils import to_dense_adj
from utils import (
    load_data,
    log_dataset_stat,
)
from scipy.sparse import csr_matrix
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--dataset', type=str, default='citeseer',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float,
                    default=0.05,  help='pertubation rate')
parser.add_argument('--model', type=str, default='PGD',
                    choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('-u', '--use_our_data', action='store_true')


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')


# Dataset
if args.use_our_data:
    dataset_name = 'Cora'
    use_gdc = False
    dataset_path = (Path(__file__) / '../../data').resolve() / dataset_name
    dataset, data = load_data(dataset_path, dataset_name, device, use_gdc)
    datastat_path = Path(__file__).parent / 'data_stat.txt'
    log_dataset_stat(data, dataset, datastat_path)
    label = data.y
    one_hot_label = F.one_hot(data.y).float()
    adj = to_dense_adj(data.edge_index)[0]
    # adj.fill_diagonal_(1)
    adj, features, labels = csr_matrix(
        adj.cpu().detach()), csr_matrix(data.x.cpu()), data.y.cpu()
    # features = normalize_feature(features)
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
else:
    data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    # features = normalize_feature(features)
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

perturbations = int(args.ptb_rate * (adj.sum()//2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
print('Adj', adj)
print('Edge num', adj.sum())
# Setup Victim Model
victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
                   dropout=0.5, weight_decay=5e-4, device=device)

victim_model = victim_model.to(device)
victim_model.fit(features, adj, labels, idx_train)

# Setup Attack Model

model = PGDAttack(model=victim_model,
                  nnodes=adj.shape[0], loss_type='CE', device=device)

model = model.to(device)


def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train)  # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()


def main():
    global features, adj, labels, idx_train, perturbations
    model.attack(features, adj, labels, idx_train,
                 perturbations, epochs=args.epochs)
    print('=== testing GCN on original(clean) graph ===')
    test(adj)
    modified_adj = model.modified_adj
    adj = adj.cuda()
    test(modified_adj)
    print('victim original_adj', adj)
    print('modified_adj', modified_adj)
    print('diff sum', (modified_adj != adj.cuda()).sum())

    diff_idx = (modified_adj != adj.cuda()).nonzero(as_tuple=False)
    # print('diff', diff_idx)
    for i, (row, col) in enumerate(diff_idx):
        if i < 10:
            print('Different index: (', str(row.item())+',', str(col.item())+')', 'ori', adj[row][col].item(
            ), '->', modified_adj[row][col].item())
    # modified_features = model.modified_features

    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./', name=f'mod_adj')
    # model.save_features(root='./', name='mod_features')


if __name__ == '__main__':
    main()
