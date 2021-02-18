from torch_geometric.utils import to_dense_adj
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os

def decorated_with(filename):
    '''filename is the file where output will be written'''
    def decorator(func):
        # '''func is the function you are "overriding", i.e. wrapping'''
        def wrapper(*args,**kwargs):
            '''*args and **kwargs are the arguments supplied 
            to the overridden function'''
            #use with statement to open, write to, and close the file safely if not os.path.exists(filename):
            mode = 'a' if os.path.exists(filename) else 'w'
            with open(filename, mode) as outputfile:
                if args:
                    outputfile.write(*args,**kwargs)
                outputfile.write('\n')
            #now original function executed with its arguments as normal
            return func(*args,**kwargs)
        wrapper.original_print = func
        return wrapper
    return decorator


def safe_remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
    else:
        # print("Can not delete the file as it doesn't exists")
        pass

def log_dataset_stat(dataset, data, filename):
    global print
    safe_remove_file(filename)
    log = decorated_with(filename)(print)

    log('Dataset Statistics'+str('='*40))
    log(f'Dataset: {dataset}:')
    log('===========================================================================================================')
    log(f'Number of graphs: {len(dataset)}')
    log(f'Number of features: {dataset.num_features}')
    log(f'Number of classes: {dataset.num_classes}')

    log(f'Data: {data}')
    log('===========================================================================================================')

    # Gather some statistics about the graph.
    log(f'Number of nodes: {data.num_nodes}')
    log(f'Number of edges: {data.num_edges}')
    log(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    log(f'Number of training nodes: {data.train_mask.sum()}')
    log(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    log(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    log(f'Contains self-loops: {data.contains_self_loops()}')
    log(f'Is undirected: {data.is_undirected()}')
    log(f'Edge index: {data.edge_index} {data.edge_index.shape}')


def log_label_relation(data, filename):
    global print
    safe_remove_file(filename)
    log = decorated_with(filename)(print)

    log('Label Relation'+str('='*40))
    A = to_dense_adj(data.edge_index)[0]
    A.fill_diagonal_(1)
    gold_Y = F.one_hot(data.y).float()
    gold_A = torch.matmul(gold_Y, gold_Y.T)

    sorted_gold_Y, sorted_Y_indices = torch.sort(data.y, descending=False)
    sorted_gold_Y = F.one_hot(sorted_gold_Y).float()
    sorted_gold_A = torch.matmul(sorted_gold_Y, sorted_gold_Y.T)

    log(f'Original A: {A}')
    log(f'Shape: {A.shape}\n')
    log(f'Gold Label Y: {gold_Y}')
    log(f'Shape: {gold_Y.shape}\n')
    log(f'Transpose of Y: {gold_Y.T}')
    log(f'Shape: {gold_Y.T.shape}\n')
    log(f'Gold A: {gold_A}')
    log(f'Shape: {gold_A.shape}\n')
    log(f'Sorted gold Y: {sorted_gold_Y}')
    log(f'Shape: {sorted_gold_Y.shape}\n')
    log(f'Sorted gold Y indices: {sorted_Y_indices}')
    log(f'Shape: {sorted_Y_indices.shape}\n')
    log(f'Gold A: {sorted_gold_A}')
    log(f'Shape: {sorted_gold_A.shape}')

def log_training(log_text, filename):
    global print
    log = decorated_with(filename)(print)
    log(log_text)

def save_model(model, edge_index, edge_weight, final, filename):
    print('Saving model')
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    checkpoint['edge_index'] = edge_index
    checkpoint['edge_weight'] = edge_weight
    checkpoint['final'] = final
    torch.save(checkpoint, filename)

    print('Saved as', filename)

def log_model_architecture(model, optimizer, filename):
    global print
    log = decorated_with(filename)(print)
    log(f'Model: {model}')
    log(f'Optimizer: {optimizer}')
    log(f'\n')

def compare_topology(pred_A, data, log_filename, fig_filename):
    global print
    safe_remove_file(log_filename)
    log = decorated_with(log_filename)(print)

    print('Confusion Matrix '+str('='*40))
    gold_Y = F.one_hot(data.y).float()
    gold_A = torch.matmul(gold_Y, gold_Y.T)
    
    flat_pred_A = pred_A.detach().cpu().view(-1)
    flat_gold_A = gold_A.detach().cpu().view(-1)
    conf_mat = confusion_matrix(y_true=flat_gold_A, y_pred=flat_pred_A)
    tn, fp, fn, tp = conf_mat.ravel()
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    f1 = 2*(ppv*tpr)/(ppv+tpr)

    log(f'Flatten A: {flat_pred_A}')
    log(f'Shape: {flat_pred_A.shape}')
    log(f'Number of Positive Prediction: {flat_pred_A.sum()} ({flat_pred_A.sum().true_divide(len(flat_pred_A))})')
    log(f'Flatten Gold A: {flat_gold_A}')
    log(f'Shape: {flat_gold_A.shape}')
    log(f'Number of Positive Class: {flat_gold_A.sum()} ({flat_gold_A.sum().true_divide(len(flat_gold_A))})')
    log(f'Confusion matrix: {conf_mat}')
    log(f'Raveled Confusion Matrix: {conf_mat.ravel()}')
    log(f'True positive: {tp} # 1 -> 1')
    log(f'False positive: {fp} # 0 -> 1')
    log(f'True negative: {tn} # 0 -> 0')
    log(f'False negative: {fn} # 1 -> 0')
    log(f'Precision: {round(ppv,5)} # TP/(TP+FP)')
    log(f'Negative predictive value: {round(npv,5)} # TN/(TN+FN)')
    log(f'Recall: {round(tpr,5)} # TP/P')
    log(f'Selectivity: {round(tnr,5)} # TN/N')
    log(f'F1 score: {f1}')

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=[0,1])
    disp.plot(values_format='d')
    plt.savefig(fig_filename)
    plt.clf()
    plt.cla()
    plt.close()


def plot_tsne(tsne_x, tsne_y, figname, label_names=None):
    print('Plot TSNE'+str('='*40))
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
        plt.scatter(X_2d[tsne_y == i, 0], X_2d[tsne_y == i, 1], c=c, s=2, label=label)
    plt.legend()
    plt.savefig(figname)
    plt.clf()
    plt.cla()
    plt.close()

    print('Saved as', figname)


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
    print('Plot Topology'+str('='*40))
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

    print('Saved as', figname)


def zero_to_nan(adj):
    nan_adj = np.copy(adj).astype('float')
    nan_adj[nan_adj==0] = np.nan # Replace every 0 with 'nan'
    return nan_adj


def plot_sorted_topology_with_gold_topology(adj, gold_adj, data, figname, sorting=False):
    print('Plot Sorted Topology With Gold Topology'+str('='*40))
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
    print('Saved as', figname)

def superprint(text, log_filename, append=False):
    global print
    if append:
        safe_remove_file(log_filename)
    log = decorated_with(log_filename)(print)
    log(text)