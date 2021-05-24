import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import torch
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.utils.sparse import dense_to_sparse
import pickle
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix, csc_matrix 

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)




def sci_sparse_to_torch(coo):

    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

def load_random_adj(dataset_str, device):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("/data/brandon/toptimize/text_data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    train_idx_orig = parse_index_file(
        "/data/brandon/toptimize/text_data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig) # TODO 트레인 인덱스가 미리 정해진것 같지만 일단은 무시한다.

    val_size = train_size - x.shape[0] # 335
    test_size = tx.shape[0] # 4043
    
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)
    doc_size = len(idx_train) + len(idx_val) + len(idx_test)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    doc_idx = np.array(list(idx_train) + list(idx_val)+list(idx_test))
    word_idx = np.array(range(len(y) + val_size, allx.shape[0]))
    # x_list = pickle.load(open('/data/brandon/BertGCN2/graph_features_ohsumed.pkl', 'rb'))
    x_list = pickle.load(open('/data/brandon/BertGCN2/graph_logits_ohsumed.pkl', 'rb'))
    x = torch.stack([sample for batch in x_list for sample in batch])
    x = x.view(-1, 23).float().to(device)
    # x = x.view(-1, 768).float().to(device)

    one_hot = labels[doc_idx][:]
    y = torch.LongTensor(np.argmax(one_hot, axis=1)).to(device)

    y_train = torch.LongTensor(y_train)

    def cold_start(edge_index, edge_attr, ratio=1):
        mask = torch.rand((1, edge_index.size(1)))[0]
        mask = mask <= ratio
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr

    adj = torch.ones((len(labels), len(labels)))
    print(adj, adj.shape)
    mask = torch.rand(21557, 21557)
    # print(mask)
    mask = mask <= 0.00001 # 4억 -> 4000개 -> 1/십만
    # print(mask)
    adj = adj * mask

    print(adj, adj.shape)
    # input()

    edge_index, edge_attr = dense_to_sparse(adj)
    edge_index = edge_index.long().to(device)
    edge_attr = edge_attr.float().to(device)

    train_mask = torch.BoolTensor(train_mask[doc_idx])
    val_mask = torch.BoolTensor(val_mask[doc_idx])
    test_mask = torch.BoolTensor(test_mask[doc_idx])

    return x, y, edge_index, edge_attr, train_mask, val_mask, test_mask

def load_empty_adj(dataset_str, device):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("/data/brandon/toptimize/text_data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    train_idx_orig = parse_index_file(
        "/data/brandon/toptimize/text_data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig) # TODO 트레인 인덱스가 미리 정해진것 같지만 일단은 무시한다.

    val_size = train_size - x.shape[0] # 335
    test_size = tx.shape[0] # 4043
    
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)
    doc_size = len(idx_train) + len(idx_val) + len(idx_test)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    doc_idx = np.array(list(idx_train) + list(idx_val)+list(idx_test))
    word_idx = np.array(range(len(y) + val_size, allx.shape[0]))
    # x_list = pickle.load(open('/data/brandon/BertGCN2/graph_features_ohsumed.pkl', 'rb'))
    x_list = pickle.load(open('/data/brandon/BertGCN2/graph_logits_ohsumed.pkl', 'rb'))
    x = torch.stack([sample for batch in x_list for sample in batch])
    x = x.view(-1, 23).float().to(device)
    # x = x.view(-1, 768).float().to(device)

    one_hot = labels[doc_idx][:]
    y = torch.LongTensor(np.argmax(one_hot, axis=1)).to(device)

    y_train = torch.LongTensor(y_train)
    adj = torch.zeros((len(labels), len(labels)))
    adj[1][24] = 1

    edge_index, edge_attr = dense_to_sparse(adj)
    edge_index = edge_index.long().to(device)
    edge_attr = edge_attr.float().to(device)

    train_mask = torch.BoolTensor(train_mask[doc_idx])
    val_mask = torch.BoolTensor(val_mask[doc_idx])
    test_mask = torch.BoolTensor(test_mask[doc_idx])

    return x, y, edge_index, edge_attr, train_mask, val_mask, test_mask

def load_lbl_agr(dataset_str, device):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("/data/brandon/toptimize/text_data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    # print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape, adj.shape)
    # (3022, 300) (3022, 23) (4043, 300) (4043, 23) (17514, 300) (17514, 23) (21557, 21557)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    # print(len(labels)) # 21557

    train_idx_orig = parse_index_file(
        "/data/brandon/toptimize/text_data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig) # TODO 트레인 인덱스가 미리 정해진것 같지만 일단은 무시한다.

    val_size = train_size - x.shape[0] # 335
    test_size = tx.shape[0] # 4043
    
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    doc_idx = np.array(list(idx_train) + list(idx_val)+list(idx_test))
    word_idx = np.array(range(len(y) + val_size, allx.shape[0]))
    x_list = pickle.load(open('/data/brandon/BertGCN2/graph_logits_ohsumed.pkl', 'rb')) # 7400 * 32
    x = torch.stack([sample for batch in x_list for sample in batch])
    x = x.view(-1, labels.shape[1]).float().to(device)

    one_hot = labels[doc_idx][:]
    y = torch.LongTensor(np.argmax(one_hot, axis=1)).to(device)

    y_train = torch.LongTensor(y_train)
    adj = torch.matmul(y_train, y_train.T)
    adj.fill_diagonal_(0)

    edge_index, edge_attr = dense_to_sparse(adj)
    edge_index = edge_index.long().to(device)
    edge_attr = edge_attr.float().to(device)

    train_mask = torch.BoolTensor(train_mask[doc_idx])
    val_mask = torch.BoolTensor(val_mask[doc_idx])
    test_mask = torch.BoolTensor(test_mask[doc_idx])

    return x, y, edge_index, edge_attr, train_mask, val_mask, test_mask


def load_bert_gnn(dataset_str, device):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("/data/brandon/toptimize/text_data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape, adj.shape)
    # (3022, 300) (3022, 23) (4043, 300) (4043, 23) (17514, 300) (17514, 23) (21557, 21557)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    print(len(labels)) # 21557

    train_idx_orig = parse_index_file(
        "/data/brandon/toptimize/text_data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig) # TODO 트레인 인덱스가 미리 정해진것 같지만 일단은 무시한다.

    val_size = train_size - x.shape[0] # 335
    test_size = tx.shape[0] # 4043
    
    idx_train = range(len(y))#; print(len(y))
    idx_val = range(len(y), len(y) + val_size) #; print(len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size);  #print(allx.shape[0] + test_size)
    # idx_test = range(len(y) + val_size, len(y) + val_size + test_size)#; print(allx.shape[0] + test_size)
    doc_size = len(idx_train) + len(idx_val) + len(idx_test)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]


    doc_idx = np.array(list(idx_train) + list(idx_val)+list(idx_test))
    print(min(idx_train), max(idx_train), len(y)) # train_size는 3357 아마 논문처럼 이어야 하지만, 코드는 3022를 사용한다.
    print(min(idx_val), max(idx_val), val_size)
    print(min(idx_test), max(idx_test), test_size)
    print(allx.shape[0], tx.shape[0])
    word_idx = np.array(range(len(y) + val_size, allx.shape[0]))
    print(doc_idx, len(doc_idx))
    print(word_idx, len(word_idx))
    # print(f'x{features.shape}')
    # x = features[:doc_size, ]
    x_list = pickle.load(open('/data/brandon/BertGCN2/graph_logits_ohsumed.pkl', 'rb')) # 7400 * 32
    # print(len(x_list), len(x_list[-1]), len(x_list[-1][-1]), type( x_list))#, x_list[-1][-1], type(x_list[-1][-1])
    x = torch.stack([sample for batch in x_list for sample in batch])
    # x = x.view(-1, 768).to(device)
    x = x.view(-1, labels.shape[1]).float().to(device)
    # print(f'x{x.shape}')

    # print(f'y{labels}, {type(labels)}')
    # one_hot1 = labels[:len(idx_train) + len(idx_val), ]
    # one_hot2 = labels[-len(idx_test):]
    # one_hot = np.concatenate((one_hot1, one_hot2))
    one_hot = labels[doc_idx][:]
    # print(f'y{one_hot.shape}')
    y = torch.LongTensor(np.argmax(one_hot, axis=1)).to(device)
    # print(f'y{y.shape}')
    # input()
    # y_np = np.argmax(one_hot, axis=1)
    # i, = np.where(y_np == 2)
    # print(i)
    # print((y == 2).nonzero(as_tuple=True)[0][0])



    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.todense() # scipy to numpy

    doc2word, word2doc = np.zeros((len(labels), len(labels))), np.zeros((len(labels), len(labels)))
    # doc2word[doc_idx, word_idx] = adj[np.ix_(doc_idx,word_idx)]
    doc2word[doc_idx[:,None], word_idx] = adj[doc_idx[:,None], word_idx]
    word2doc[word_idx[:,None], doc_idx]= adj[word_idx[:,None], doc_idx]
    adj = adj + doc2word.dot(word2doc)

    print(len(np.where(adj>0)[0]))
    # adj = np.delete(adj, np.s_[len(idx_train) + len(idx_val):-len(idx_test)], axis=1)
    # adj = np.delete(adj, np.s_[len(idx_train) + len(idx_val):-len(idx_test)], axis=0)
    # print(list(idx_train) + list(idx_val)+list(idx_test))
    # indices = np.array(list(idx_train) + list(idx_val)+list(idx_test))

    adj = adj[doc_idx[:,None],doc_idx]
    print(len(np.where(adj>0)[0]))
    print(adj, adj.shape)
    print(np.where(adj>0))
    print(adj, adj.shape)
    # adj[1][24] = 1
 
    # adj = csr_matrix(adj)
    # print(adj)

    # adj = sci_sparse_to_torch(adj)
    # print(f'adj{adj.shape}')
    # adj = adj[:doc_size, :doc_size]
    
    # print(f'shr{adj.shape}')
    edge_index, edge_attr = dense_to_sparse(torch.from_numpy(adj))
    # edge_index, edge_attr = from_scipy_sparse_matrix(adj)
    edge_index = edge_index.long().to(device)
    edge_attr = edge_attr.float().to(device)

    # edge_attr = edge_attr
    # edge_index = edge_index
    # print(train_mask[:doc_size], len(train_mask[:doc_size]))
    # print(val_mask[:doc_size], len(val_mask[:doc_size]))
    # print(test_mask[:doc_size], len(test_mask[:doc_size]))
    # input()

    train_mask = torch.BoolTensor(train_mask[doc_idx])
    val_mask = torch.BoolTensor(val_mask[doc_idx])
    test_mask = torch.BoolTensor(test_mask[doc_idx])

    return x, y, edge_index, edge_attr, train_mask, val_mask, test_mask
    # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


# x, y, edge_index, edge_attr, train_mask, val_mask, test_mask = load_bertgcn('ohsumed')



# print(features.shape)
# values = features.data
# indices = np.vstack((features.row, features.col))

# i = torch.LongTensor(indices)
# v = torch.FloatTensor(values)
# shape = coo.shape

# features = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
# print(features.shape)

# features,  = from_scipy_sparse_matrix(features)
