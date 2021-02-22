from torch_geometric.utils import to_dense_adj
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from shutil import rmtree
import matplotlib


def load_data(data_path, dataset_name, device, use_gdc):
    # Data
    dataset = Planetoid(data_path, dataset_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    if use_gdc:
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
                    sparsification_kwargs=dict(method='topk', k=128,
                                               dim=0), exact=True)
        data = gdc(data)
    return dataset, data


def decorated_with(filename):
    '''filename is the file where output will be written'''
    def decorator(func):
        # '''func is the function you are "overriding", i.e. wrapping'''
        def wrapper(*args, **kwargs):
            '''*args and **kwargs are the arguments supplied
            to the overridden function'''
            # use with statement to open, write to, and close the file safely if not os.path.exists(filename):
            mode = 'a' if os.path.exists(filename) else 'w'
            with open(filename, mode) as outputfile:
                if args:
                    outputfile.write(*args, **kwargs)
                outputfile.write('\n')
            # now original function executed with its arguments as normal
            return func(*args, **kwargs)
        wrapper.original_print = func
        return wrapper
    return decorator


def safe_remove_file(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)
    else:
        pass


def safe_remove_dir(dirpath):
    if os.path.exists(dirpath):
        rmtree(dirpath)
    else:
        pass


def evaluate_experiment(step, final, label, adj, gold_adj, confmat_dir, topofig_dir, tsne_dir, prev_stat=None):

    perf_stat = compare_topology(
        adj, gold_adj, confmat_dir/('confmat'+str(step)+'.txt'), confmat_dir/('confmat'+str(step)+'.png'))
    if prev_stat:
        tp_gain = perf_stat['tp'] - prev_stat['tp']
        fp_gain = perf_stat['fp'] - prev_stat['fp']
        perf_stat['step'] = step
        perf_stat['tp_gain'] = tp_gain
        perf_stat['fp_gain'] = fp_gain
        superprint(f"TP Gain: {perf_stat['tp_gain']}",
                   confmat_dir / ('confmat'+str(step)+'.txt'))
        superprint(f"FP Gain: {perf_stat['fp_gain']}",
                   confmat_dir / ('confmat'+str(step)+'.txt'))
        try:
            perf_stat['tp_over_fp'] = round(tp_gain / fp_gain, 4)
            if perf_stat['tp_over_fp'] > 0:
                superprint(f"Ratio: {perf_stat['tp_over_fp']}",
                           confmat_dir / ('confmat'+str(step)+'.txt'))
        except:
            pass

    crossplot_topology(adj, gold_adj, label, topofig_dir /
                       ('topofig'+str(step)+'.png'))
    plot_tsne(final, label, tsne_dir/('tsne_'+str(step)+'.png'))

    return perf_stat


def log_dataset_stat(dataset, filename):
    global print
    safe_remove_file(filename)
    log = decorated_with(filename)(print)

    log('Dataset Statistics'+str('='*40))
    log(f'Dataset: {dataset}:')
    log('===========================================================================================================')
    log(f'Number of graphs: {len(dataset)}')
    log(f'Number of features: {dataset.num_features}')
    log(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]

    log(f'Data 0: {data}')
    log('===========================================================================================================')

    # Gather some statistics about the graph.
    log(f'Number of nodes: {data.num_nodes}')
    log(f'Number of edges: {data.num_edges}')
    log(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    labeled_node_num = int(data.train_mask.sum() +
                           data.val_mask.sum()+data.test_mask.sum())
    log(f'Number of labeled nodes: {labeled_node_num}')
    log(f'Number of training nodes: {data.train_mask.sum()}')
    log(f'Number of validation nodes: {data.val_mask.sum()}')
    log(f'Number of test nodes: {data.test_mask.sum()}')
    log(f'Node label rate: {int(labeled_node_num) / data.num_nodes:.2f}')
    log(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    log(f'Validation node label rate: {int(data.val_mask.sum()) / data.num_nodes:.2f}')
    log(f'Test node label rate: {int(data.test_mask.sum()) / data.num_nodes:.2f}')
    log(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    log(f'Contains self-loops: {data.contains_self_loops()}')
    log(f'Is undirected: {data.is_undirected()}')
    log(f'Edge index: {data.edge_index} {data.edge_index.shape}')
    log(f'Edge weight: {data.edge_attr}')


def log_training(log_text, filename, overwrite=False):
    global print
    log = decorated_with(filename)(print)

    if overwrite:
        safe_remove_file(filename)
    log(log_text)


def log_model_architecture(step, model, optimizer, filename, overwrite=False):
    global print

    if overwrite:
        safe_remove_file(filename)
    log = decorated_with(filename)(print)
    log(f'Model Architecture {step} {"="*40}')
    log(f'Model: {model}')
    log(f'Optimizer: {optimizer}')
    log(f'\n')


def percentage(float_zero_to_one):
    return round(float_zero_to_one * 100, 2)


def compare_topology(in_adj, gold_adj, log_filename, fig_filename):
    global print
    safe_remove_file(log_filename)
    log = decorated_with(log_filename)(print)

    print('Confusion Matrix '+str('='*40))

    adj = in_adj.clone()
    adj.fill_diagonal_(1)

    flat_adj = adj.detach().cpu().view(-1)
    flat_gold_adj = gold_adj.detach().cpu().view(-1)
    conf_mat = confusion_matrix(y_true=flat_gold_adj, y_pred=flat_adj)

    tn, fp, fn, tp = conf_mat.ravel()
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    f1 = 2*((ppv*tpr)/(ppv+tpr))

    ppv = percentage(ppv)
    npv = percentage(npv)
    tpr = percentage(tpr)
    tnr = percentage(tnr)
    f1 = percentage(f1)

    log(f'True Positive: {tp}')
    log(f'False Positive: {fp}')
    log(f'True Negative: {tn}')
    log(f'False Negative: {fn}')
    log(f'Precision: {ppv}% # TP/(TP+FP)')
    log(f'Negative Predictive Value: {npv}% # TN/(TN+FN)')
    log(f'Recall: {tpr}% # TP/P')
    log(f'Selectivity: {tnr}% # TN/N')
    log(f'F1 Score: {f1}%')
    # log(f'Confusion matrix: {conf_mat}')
    # log(f'Raveled Confusion Matrix: {conf_mat.ravel()}')

    edge_sum = flat_adj.sum()
    gold_sum = flat_gold_adj.sum()
    edge_ratio = round(float(edge_sum.div(gold_sum))*100, 2)
    log(f'# Edge: {int(edge_sum)}')
    log(f'# Gold Edge: {int(gold_sum)}')
    log(f'# Edge over # Gold Edge: {edge_ratio}%')
    disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=[0, 1])
    matplotlib.use('Agg')
    disp.plot(values_format='d')

    plt.savefig(fig_filename)
    plt.clf()
    plt.cla()
    plt.close()

    result = {'tp': tp, 'fp': fp,
              'tn': tn, 'fn': fn,
              'ppv': ppv, 'tpr': tpr,
              'npv': npv, 'tnr': tnr,
              'f1': f1, 'edge num': tp + fp}
    return result


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
        plt.scatter(X_2d[tsne_y == i, 0],
                    X_2d[tsne_y == i, 1], c=c, s=2, label=label)
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
    adj = adj[sorted_Y_indices]  # sort row
    adj = adj[:, sorted_Y_indices]  # sort column
    return adj


def plot_topology(in_adj, data, figname, sorting=True):
    print('Plot Topology'+str('='*40))

    adj = in_adj.clone()
    adj.fill_diagonal_(1)

    from matplotlib import pyplot as plt
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(3000.0/float(DPI), 3000.0/float(DPI))

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
    nan_adj[nan_adj == 0] = np.nan  # Replace every 0 with 'nan'
    return nan_adj


def crossplot_topology(in_adj, in_gold_adj, in_label, figname, sorting=True):
    print('Crossplot Topology'+str('='*40))

    adj = in_adj.clone()
    gold_adj = in_gold_adj.clone()
    label = in_label.clone()
    adj.fill_diagonal_(1)

    from matplotlib import pyplot as plt
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(3000.0/float(DPI), 3000.0/float(DPI))

    adj = cpu(adj)
    adj = numpy(adj)
    gold_adj = cpu(gold_adj)
    gold_adj = numpy(gold_adj)

    if sorting:
        sorted_gold_Y, sorted_Y_indices = torch.sort(label, descending=False)
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


def superprint(text, log_filename, overwrite=False):
    global print
    if overwrite:
        safe_remove_file(log_filename)
    log = decorated_with(log_filename)(print)
    log(text)


def cold_start():

    # print('data.edge_index', data.edge_index, data.edge_index.shape)
    # mask = torch.randint(0, 3 + 1, (1, data.edge_index.size(1)))[0]
    # print('mask', mask, mask.shape)
    # mask = mask >= 3
    # print('mask', mask, mask.shape)
    # data.edge_index = data.edge_index[:, mask]
    # print('data.edge_index', data.edge_index, data.edge_index.shape)
    pass


def log_hyperparameters(args, hyper_path):
    # Logging hyperparameters
    safe_remove_file(hyper_path)
    for key, val in args.__dict__.items():
        superprint(f'{key}: {val}', hyper_path)


def log_step_perf(val_accs, test_accs, filename):

    val_accs = np.array(val_accs)
    mean = round(np.mean(val_accs), 2)
    std = round(np.std(val_accs), 2)

    test_accs = np.array(test_accs)
    mean = round(np.mean(test_accs), 2)
    std = round(np.std(test_accs), 2)

    superprint(f'Step Performance Summary {"="*40}', filename, overwrite=True)
    superprint(f'Final Val. Acc.: {val_accs[-1]}', filename)
    superprint(f'Final Test. Acc.: {test_accs[-1]}', filename)
    superprint(f'Mean Val Acc: {mean} +/- {std}', filename)
    superprint(f'Mean Test Acc: {mean} +/- {std}', filename)
    superprint(f'Vals Accs: {val_accs}', filename)
    superprint(f'Test Accs {test_accs}', filename)


def log_run_perf(base_vals, base_tests, ours_vals, ours_tests, filename):
    superprint(
        f'Run Performance Comparision {"="*40}', filename, overwrite=True)

    val_accs = np.array(base_vals)
    mean = round(np.mean(base_vals), 2)
    std = round(np.std(base_vals), 2)

    test_accs = np.array(base_tests)
    mean = round(np.mean(base_tests), 2)
    std = round(np.std(base_tests), 2)

    superprint(f'Baseline', filename, overwrite=True)
    superprint(f'Mean Val Acc: {mean} +/- {std}', filename)
    superprint(f'Mean Test Acc: {mean} +/- {std}', filename)
    superprint(f'Vals Accs: {val_accs}', filename)
    superprint(f'Test Accs {test_accs}', filename)

    val_accs = np.array(ours_vals)
    mean = round(np.mean(ours_vals), 2)
    std = round(np.std(ours_vals), 2)

    test_accs = np.array(ours_tests)
    mean = round(np.mean(ours_tests), 2)
    std = round(np.std(ours_tests), 2)

    superprint(f'Ours', filename)
    superprint(f'Mean Val Acc: {mean} +/- {std}', filename)
    superprint(f'Mean Test Acc: {mean} +/- {std}', filename)
    superprint(f'Vals Accs: {val_accs}', filename)
    superprint(f'Test Accs {test_accs}', filename)
