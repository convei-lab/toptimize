import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv  # noqa
from torch_geometric.nn import GCN4ConvSIGIR, GAT4ConvSIGIR
from torch_geometric.utils.sparse import dense_to_sparse
from utils import log_grad


class GCN(torch.nn.Module):
    def __init__(self, nfeat, hidden_sizes, nclass, cached=True, use_gdc=False, return_final=True):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = hidden_sizes
        self.nclass = nclass
        self.conv1 = GCNConv(nfeat, hidden_sizes,
                             cached=cached, normalize=not use_gdc)
        self.conv2 = GCNConv(hidden_sizes, nclass,
                             cached=cached, normalize=not use_gdc)
        if not cached:
            self.conv1._cached_edge_index = None
            self.conv1._cached_adj_t = None
            self.conv2._cached_edge_index = None
            self.conv2._cached_adj_t = None
        self.return_final = return_final

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        final = self.conv2(x, edge_index, edge_attr)
        if self.return_final:
            return final, F.log_softmax(final, dim=1)
        else:
            return F.log_softmax(final, dim=1)

class OurGCN(torch.nn.Module):
    def __init__(self, nfeat, hidden_sizes, nclass, alpha=10, beta=-3, cached=True, use_gdc=False, return_final=True):
        super(OurGCN, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = hidden_sizes
        self.nclass = nclass
        self.conv1 = GCN4ConvSIGIR(
            nfeat, hidden_sizes, cached=cached, alpha=alpha, beta=beta, normalize=not use_gdc)
        self.conv2 = GCNConv(hidden_sizes, nclass,
                             cached=cached, normalize=not use_gdc)
        self.return_final = return_final

        if not cached:
            self.conv1._cached_edge_index = None
            self.conv1._cached_adj_t = None
            self.conv2._cached_edge_index = None
            self.conv2._cached_adj_t = None

    def forward(self, x, edge_index, edge_attr=None):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        final = self.conv2(x, edge_index, edge_attr)
        if self.return_final:
            return final, F.log_softmax(final, dim=1)
        else:
            return F.log_softmax(final, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, nfeat, hidden_sizes, nclass, nhead=8, dropout=0.6, return_final=True):
        super(GAT, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = hidden_sizes
        self.nclass = nclass
        self.nhead = nhead
        self.dropout = dropout
        self.conv1 = GATConv(nfeat, hidden_sizes, heads=nhead, dropout=dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(hidden_sizes * nhead, nclass, heads=1, concat=False,
                             dropout=dropout)
        self.return_final = return_final

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        final = F.elu(self.conv2(x, edge_index))
        # final = self.conv2(x, edge_index)
        if self.return_final:
            return final, F.log_softmax(final, dim=1)
        else:
            return F.log_softmax(final, dim=1)


class OurGAT(torch.nn.Module):
    def __init__(self, nfeat, hidden_sizes, nclass, alpha=10, beta=-3, nhead=8, dropout=0.6, return_final=True):
        super(OurGAT, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = hidden_sizes
        self.nclass = nclass
        self.nhead = nhead
        self.dropout = dropout
        self.conv1 = GAT4ConvSIGIR(
            nfeat, hidden_sizes, alpha=alpha, beta=beta, heads=nhead, dropout=dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(hidden_sizes * nhead, nclass, heads=1, concat=False,
                             dropout=dropout)
        self.return_final = return_final

    def forward(self, x, edge_index, edge_attr=None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        final = F.elu(self.conv2(x, edge_index))
        if self.return_final:
            return final, F.log_softmax(final, dim=1)
        else:
            return F.log_softmax(final, dim=1)
