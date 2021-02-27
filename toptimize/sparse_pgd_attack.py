"""
    Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective
        https://arxiv.org/pdf/1906.04214.pdf
    Tensorflow Implementation:
        https://github.com/KaidiXu/GCN_ADV_Train
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm

from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack
from torch.nn.modules.module import Module


class SparsePGDAttack(Module):
    """PGD attack for graph data.
    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    loss_type: str
        attack loss type, chosen from ['CE', 'CW']
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    device: str
        'cpu' or 'cuda'
    Examples
    --------
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import PGDAttack
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    # conver to tensor
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> # Setup Victim Model
    >>> victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0.5, weight_decay=5e-4, device='cpu').to('cpu')
    >>> victim_model.fit(features, adj, labels, idx_train)
    >>> # Setup Attack Model
    >>> model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device='cpu').to('cpu')
    >>> model.attack(features, adj, labels, idx_train, n_perturbations=10)
    >>> modified_adj = model.modified_adj
    """

    def __init__(self, model=None, nnodes=None, loss_type='CE', feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):

        super(SparsePGDAttack, self).__init__()

        self.surrogate = model
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.attack_features = attack_features
        print(device)
        self.device = device
        self.modified_adj = None
        self.modified_features = None
        if model is not None:
            self.nclass = model.nclass
            self.nfeat = model.nfeat
            self.hidden_sizes = model.hidden_sizes

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.loss_type = loss_type
        self.modified_edge_index = None
        self.modified_edge_attr = None
        self.modified_features = None

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            self.edge_changes = Parameter(
                torch.FloatTensor(int(nnodes*(nnodes-1)/2)).to('cuda'))
            self.edge_changes.data.fill_(0)
            self.nnodes = nnodes

        if attack_features:
            assert True, 'Topology Attack does not support attack feature'

        self.complementary = None

    def attack(self, ori_features, ori_edge_index, ori_edge_attr, labels, idx_train, n_perturbations, epochs=200, device='cuda', **kwargs):
        """Generate perturbations on the input graph.
        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_edge_index :
            Original (unperturbed) edge index in the sparse form of (row, col)
        ori_edge_attr :
            Original (unperturbed) edge attribute in the sparse form of (col)
        labels :
            node labels
        idx_train :
            node training indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        epochs:
            number of training epochs
        """

        victim_model = self.surrogate
        victim_model.eval()
        for t in tqdm(range(epochs)):
            modified_edge_index, modified_edge_attr = self.get_modified_edge(
                ori_edge_index, ori_edge_attr, device=device)
            print('modified_edge', modified_edge_index, modified_edge_attr)
            # adj_norm = utils.normalize_adj_tensor(modified_adj)
            _final, output = victim_model(
                ori_features, modified_edge_index, modified_edge_attr)
            loss = self._loss(output[idx_train], labels[idx_train])
            loss.backward()

            print('\n'+'Model Parameter')
            for i, param in enumerate(victim_model.parameters()):
                print(f'Layer Index: {i}')
                print(f'Parameter: {param}')
                print(f'Name: {param.name}')
                print(f'Shape: {param.data.shape}')
                print(f'Requires Gradient: {param.requires_grad}')
                print(f'Gradient: {param.grad}')
                print(f'Leaf Layer: {param.is_leaf}')
                print(f'Data as Tensor: {param.data}')
                print()

            print('\n' + 'Tracing Gradient Backwards')
            def getBack(var_grad_fn):
                print(var_grad_fn)
                for n in var_grad_fn.next_functions:
                    if n[0]:
                        try:
                            tensor = getattr(n[0], 'variable')
                            print(n[0])
                            print('Tensor with grad found:', tensor)
                            print(' - gradient:', tensor.grad)
                            print(' - shape:', tensor.shape)
                            print()
                        except AttributeError as e:
                            getBack(n[0])
            getBack(loss.grad_fn)
            adj_grad = torch.autograd.grad(
                loss, self.edge_changes)[0]

            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            if self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t+1)
                self.adj_changes.data.add_(lr * adj_grad)

            self.projection(n_perturbations)

        self.random_sample(ori_adj, ori_features, labels,
                           idx_train, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        self.check_adj_tensor(self.modified_adj)

    def random_sample(self, ori_adj, ori_features, labels, idx_train, n_perturbations):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                # print(sampled.sum())
                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                # loss = F.nll_loss(output[idx_train], labels[idx_train])
                # print(loss)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss

    def projection(self, n_perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5)
            self.adj_changes.data.copy_(torch.clamp(
                self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(
                self.adj_changes.data, min=0, max=1))

    def get_modified_edge(self, ori_edge_index, ori_edge_attr, device):
        self.edge_changes.retain_grad()
        if self.complementary is None:

            with torch.autograd.detect_anomaly():
                arange_index = torch.arange(self.nnodes, device=device)
                row_index = arange_index.repeat_interleave(self.nnodes)
                col_index = arange_index.repeat(self.nnodes)
                adj_index = torch.stack([row_index, col_index])
                del row_index, col_index
                adj_attr = torch.zeros_like(
                    adj_index[0], dtype=torch.float, device=device)
                ori_agg_index = self.nnodes*ori_edge_index[0]+ori_edge_index[1]
                adj_attr[ori_agg_index] = ori_edge_attr

                # print(ori_agg_index.shape, torch.max(ori_agg_index))
                # print(ori_edge_attr.shape, ori_edge_attr.device, ori_edge_attr.dtype,
                #       torch.max(ori_edge_attr))
                # print(adj_attr.shape, adj_attr.device, adj_attr.dtype)
                # print(adj_attr[ori_agg_index].shape)
                # for test_idx in range(ori_agg_index.size(0)):
                #     # print(ori_edge_index[0][test_idx],
                #     #       ori_edge_index[1][test_idx], ori_agg_index[test_idx])
                #     # print(adj_attr[ori_agg_index[10]], ori_edge_attr[test_idx])
                #     assert adj_attr[ori_agg_index[test_idx]
                #                     ] == ori_edge_attr[test_idx]
                ones_attr = torch.ones_like(
                    adj_index[0], dtype=torch.float, device=device)
                eye_agg_index = self.nnodes*arange_index+arange_index
                # print(eye_agg_index)
                ones_attr[eye_agg_index] = 0
                ones_attr[ori_agg_index] -= 2
                # print(adj_attr)
                self.complementary = ones_attr

        m = torch.zeros((self.nnodes * self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(
            row=self.nnodes, col=self.nnodes, offset=-1)
        m[self.nnodes*tril_indices[0]+tril_indices[1]] = self.edge_changes
        m[self.nnodes*tril_indices[1]+tril_indices[0]] = self.edge_changes
        print('self.edge_changes', self.edge_changes,
              self.edge_changes.requires_grad)
        modified_attr = self.complementary * m + adj_attr

        return adj_index, modified_attr

    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations

        miu = a
        while ((b-a) >= epsilon):
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu
