import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv  # noqa
from torch_geometric.nn import GCN4ConvSIGIR, GAT4ConvSIGIR


class GCN(torch.nn.Module):
    def __init__(self, nfeat, hidden_sizes, nclass, use_gdc=False):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = hidden_sizes
        self.nclass = nclass
        self.conv1 = GCNConv(nfeat, hidden_sizes,
                             cached=True, normalize=not use_gdc)
        self.conv2 = GCNConv(hidden_sizes, nclass,
                             cached=True, normalize=not use_gdc)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        final = self.conv2(x, edge_index, edge_attr)
        return final, F.log_softmax(final, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, nfeat, hidden_sizes, nclass, nhead=8, dropout=0.6):
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

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        final = self.conv2(x, edge_index)
        return final, F.log_softmax(final, dim=1)


# class OurGCN(GCN):
#     def __init__(self, nfeat, hidden_sizes, nclass, use_gdc=False):
#         super().__init__(nfeat, hidden_sizes, nclass, use_gdc=False)
#         self.conv1 = GCN4ConvSIGIR(nfeat, hidden_sizes,
#                                    cached=True, normalize=not use_gdc)


# class OurGAT(GAT):
#     def __init__(self, nfeat, hidden_sizes, nclass, nhead=8, dropout=0.6):
#         super().__init__(nfeat, hidden_sizes, nclass, nhead=8, dropout=0.6)
#         self.conv1 = GAT4ConvSIGIR(
#             nfeat, hidden_sizes, heads=nhead, dropout=dropout)


class OurGCN(torch.nn.Module):
    def __init__(self, nfeat, hidden_sizes, nclass, alpha=10, beta=-3, cached=True, use_gdc=False):
        super(OurGCN, self).__init__()
        self.nfeat = nfeat
        self.hidden_sizes = hidden_sizes
        self.nclass = nclass
        self.conv1 = GCN4ConvSIGIR(
            nfeat, hidden_sizes, cached=cached, alpha=alpha, beta=beta, normalize=not use_gdc)
        self.conv2 = GCNConv(hidden_sizes, nclass,
                             cached=True, normalize=not use_gdc)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        final = self.conv2(x, edge_index, edge_attr)
        return final, F.log_softmax(final, dim=1)

    def reset_cached(self):
        self.conv1._cached_adj_t = None
        self.conv1._cached_edge_index = None


class OurGAT(torch.nn.Module):
    def __init__(self, nfeat, hidden_sizes, nclass, alpha=10, beta=-3, nhead=8, dropout=0.6):
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

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        final = self.conv2(x, edge_index)
        return final, F.log_softmax(final, dim=1)
