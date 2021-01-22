from torch_geometric.utils import to_dense_adj
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def print_dataset_stat(dataset, data):
    print()
    print(f'Dataset: {dataset}:')
    print('===========================================================================================================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    print()
    print(data)
    print('===========================================================================================================')

    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')
    print(f'Edge index: {data.edge_index} {data.edge_index.shape}')

def print_label_relation(data):
    A = to_dense_adj(data.edge_index)[0]
    A.fill_diagonal_(1)
    gold_Y = F.one_hot(data.y).float()
    gold_A = torch.matmul(gold_Y, gold_Y.T)

    sorted_gold_Y, sorted_Y_indices = torch.sort(data.y, descending=False)
    sorted_gold_Y = F.one_hot(sorted_gold_Y).float()
    sorted_gold_A = torch.matmul(sorted_gold_Y, sorted_gold_Y.T)

    print()
    print('Label Relation')
    print('============================================================')
    print(f'Original A: {A}')
    print(f'Shape: {A.shape}')
    print(f'Gold Label Y: {gold_Y}')
    print(f'Shape: {gold_Y.shape}')
    print(f'Transpose of Y: {gold_Y.T}')
    print(f'Shape: {gold_Y.T.shape}')
    print(f'Gold A: {gold_A}')
    print(f'Shape: {gold_A.shape}')
    print(f'Sorted gold Y: {sorted_gold_Y}')
    print(f'Shape: {sorted_gold_Y.shape}')
    print(f'Sorted gold Y indices: {sorted_Y_indices}')
    print(f'Shape: {sorted_Y_indices.shape}')
    print(f'Gold A: {sorted_gold_A}')
    print(f'Shape: {sorted_gold_A.shape}')

def compare_topology(pred_A, data, cm_filename='confusion_matrix_display'):
    gold_Y = F.one_hot(data.y).float()
    gold_A = torch.matmul(gold_Y, gold_Y.T)
    
    flat_pred_A = pred_A.detach().cpu().view(-1)
    flat_gold_A = gold_A.detach().cpu().view(-1)
    conf_mat = confusion_matrix(y_true=flat_gold_A, y_pred=flat_pred_A)
    # print('conf_mat', conf_mat, conf_mat.shape)
    # print('conf_mat.ravel()', conf_mat.ravel(), conf_mat.ravel().shape)
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
    print(f'Precision: {round(ppv,5)} # TP/(TP+FP)')
    print(f'Negative predictive value: {round(npv,5)} # TN/(TN+FN)')
    print(f'Recall: {round(tpr,5)} # TP/P')
    print(f'Selectivity: {round(tnr,5)} # TN/N')
    print(f'F1 score: {f1}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.savefig(cm_filename+'.png')
    plt.clf()
    plt.cla()
    plt.close()


def plot_tsne(tsne_x, tsne_y, fig_name, label_names=None):
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    
    if tsne_x.is_cuda:
        tsne_x = tsne_x.detach().cpu()
    if tsne_y.is_cuda:
        tsne_y = tsne_y.detach().cpu()
    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(tsne_x.cpu())

    target_ids = range(len(tsne_y))
    if not label_names:
        label_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g'] 

    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'y', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, label_names):
        plt.scatter(X_2d[tsne_y == i, 0], X_2d[tsne_y == i, 1], c=c, s=3, label=label)
    plt.legend()
    plt.savefig(fig_name)
    plt.clf()
    plt.cla()
    plt.close()

def cpu(variable):
    if hasattr(variable, 'is_cuda') and variable.is_cuda:
        variable = variable.detach().cpu()
    return variable

def numpy(variable):
    if not isinstance(variable, np.ndarray):
        variable = variable.numpy()
    return variable

def sort_topology(adj, sorted_Y_indices):
    sorted_Y_indices = sorted_Y_indices.cpu().numpy()
    adj = adj[sorted_Y_indices] # sort row
    adj = adj[:, sorted_Y_indices] # sort column
    return adj

def plot_topology(adj, data, figname, sorting=False):
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(3000.0/float(DPI),3000.0/float(DPI))

    adj = cpu(adj)
    adj = numpy(adj)
    if sorting:
        sorted_gold_Y, sorted_Y_indices = torch.sort(data.y, descending=False)
        adj = sort_topology(adj, sorted_Y_indices)

    plt.imshow(adj, cmap='hot', interpolation='none')    

    plt.savefig(figname)
    plt.clf()
    plt.cla()
    plt.close()

def zero_to_nan(adj):
    nan_adj = np.copy(adj).astype('float')
    nan_adj[nan_adj==0] = np.nan # Replace every 0 with 'nan'
    return nan_adj

def plot_sorted_topology_with_gold_topology(adj, gold_adj, data, figname, sorting=False):
    from matplotlib import pyplot as plt
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(3000.0/float(DPI),3000.0/float(DPI))

    adj = cpu(adj)
    adj = numpy(adj)
    gold_adj = cpu(gold_adj)
    gold_adj = numpy(gold_adj)

    if sorting:
        sorted_gold_Y, sorted_Y_indices = torch.sort(data.y, descending=False)
        adj = sort_topology(adj, sorted_Y_indices)
        gold_adj = sort_topology(gold_adj, sorted_Y_indices)

    plt.imshow(gold_adj, cmap='hot', interpolation='none')

    intra_edge_mask = gold_adj == 1
    intra_edges = adj * intra_edge_mask
    # print('intra_edges', intra_edges, intra_edges.shape, np.sum(intra_edges))

    intra_edges = zero_to_nan(intra_edges)
    plt.imshow(intra_edges, cmap='bwr', interpolation='none')
    
    inter_edge_mask = ~intra_edge_mask
    inter_edges = adj * inter_edge_mask
    # print('inter_edges', inter_edges, inter_edges.shape, np.sum(inter_edges))

    inter_edges = zero_to_nan(inter_edges)
    plt.imshow(inter_edges, cmap='hsv', interpolation='none')
    
    plt.savefig(figname)
    plt.clf()
    plt.cla()
    plt.close()