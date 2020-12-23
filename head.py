import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import sys

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

print(dataset)

data = dataset[0]
print(data)

e = data.edge_index
print(e)
