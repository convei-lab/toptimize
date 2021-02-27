from torch._C import dtype
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.sparse import dense_to_sparse
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
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import PGDAttack
from sparse_pgd_attack import SparsePGDAttack
from deeprobust.graph.utils import add_self_loops, preprocess
from scipy.sparse import csr_matrix
from torch_geometric.nn import GCN4ConvSIGIR, GAT4ConvSIGIR


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


def log_dataset_stat(data, dataset, filename):
    global print
    safe_remove_file(filename)
    log = decorated_with(filename)(print)

    log('Dataset Statistics'+str('='*40))
    log(f'Dataset: {dataset}:')
    log('===========================================================================================================')
    log(f'Number of graphs: {len(dataset)}')
    log(f'Number of features: {dataset.num_features}')
    log(f'Number of classes: {dataset.num_classes}')

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


def compare_topology(in_adj, gold_adj, log_filename, fig_filename, add_loop=True):
    global print
    safe_remove_file(log_filename)
    log = decorated_with(log_filename)(print)

    print('Confusion Matrix '+str('='*40))

    adj = in_adj.clone()
    if add_loop:
        adj.fill_diagonal_(1)
    gold_adj = gold_adj.clone()

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


def cold_start(edge_index, ratio=1):

    mask = torch.rand((1, edge_index.size(1)))[0]
    mask = mask <= ratio
    edge_index = edge_index[:, mask]
    return edge_index


def log_hyperparameters(args, hyper_path):
    # Logging hyperparameters
    safe_remove_file(hyper_path)
    for key, val in args.__dict__.items():
        superprint(f'{key}: {val}', hyper_path)


def log_step_perf(val_accs, test_accs, noen_vals, noen_tests, filename):

    val_accs = np.array(val_accs)
    mean = round(np.mean(val_accs), 2)
    std = round(np.std(val_accs), 2)

    test_accs = np.array(test_accs)
    mean = round(np.mean(test_accs), 2)
    std = round(np.std(test_accs), 2)

    superprint(
        f'Step Performance Summary (Ensemble) {"="*40}', filename, overwrite=True)
    superprint(f'Final Val. Acc.: {val_accs[-1]}', filename)
    superprint(f'Final Test. Acc.: {test_accs[-1]}', filename)
    superprint(f'Mean Val Acc: {mean} +/- {std}', filename)
    superprint(f'Mean Test Acc: {mean} +/- {std}', filename)
    superprint(f'Vals Accs: {val_accs}', filename)
    superprint(f'Test Accs {test_accs}', filename)

    val_accs = np.array(noen_vals)
    mean = round(np.mean(noen_vals), 2)
    std = round(np.std(noen_vals), 2)

    test_accs = np.array(noen_tests)
    mean = round(np.mean(noen_tests), 2)
    std = round(np.std(noen_tests), 2)

    superprint(
        f'Step Performance Summary (No Ensemble){"="*40}', filename)
    superprint(f'Final Val. Acc.: {val_accs[-1]}', filename)
    superprint(f'Final Test. Acc.: {test_accs[-1]}', filename)
    superprint(f'Mean Val Acc: {mean} +/- {std}', filename)
    superprint(f'Mean Test Acc: {mean} +/- {std}', filename)
    superprint(f'Vals Accs: {val_accs}', filename)
    superprint(f'Test Accs {test_accs}', filename)


def log_run_perf(base_vals, base_tests, ours_vals, ours_tests, noen_our_vals, noen_our_tests, filename):
    superprint(
        f'Run Performance Comparision {"="*40}', filename, overwrite=True)

    val_accs = np.array(base_vals)
    mean_val = round(np.mean(base_vals), 2)
    std_val = round(np.std(base_vals), 2)

    test_accs = np.array(base_tests)
    mean_test = round(np.mean(base_tests), 2)
    std_test = round(np.std(base_tests), 2)

    superprint(f'Baseline', filename, overwrite=True)
    superprint(f'Mean Val Acc: {mean_val} +/- {std_val}', filename)
    superprint(f'Mean Test Acc: {mean_test} +/- {std_test}', filename)
    superprint(f'Vals Accs: {val_accs}', filename)
    superprint(f'Test Accs {test_accs}', filename)

    val_accs = np.array(ours_vals)
    mean_val = round(np.mean(ours_vals), 2)
    std_val = round(np.std(ours_vals), 2)

    test_accs = np.array(ours_tests)
    mean_test = round(np.mean(ours_tests), 2)
    std_test = round(np.std(ours_tests), 2)

    superprint(f'Ours', filename)
    superprint(f'Mean Val Acc: {mean_val} +/- {std_val}', filename)
    superprint(f'Mean Test Acc: {mean_test} +/- {std_test}', filename)
    superprint(f'Vals Accs: {val_accs}', filename)
    superprint(f'Test Accs {test_accs}', filename)

    val_accs = np.array(noen_our_vals)
    mean_val = round(np.mean(noen_our_vals), 2)
    std_val = round(np.std(noen_our_vals), 2)

    test_accs = np.array(noen_our_tests)
    mean_test = round(np.mean(noen_our_tests), 2)
    std_test = round(np.std(noen_our_tests), 2)

    superprint(f'No Ensemble Ours', filename)
    superprint(f'Mean Val Acc: {mean_val} +/- {std_val}', filename)
    superprint(f'Mean Test Acc: {mean_test} +/- {std_test}', filename)
    superprint(f'Vals Accs: {val_accs}', filename)
    superprint(f'Test Accs {test_accs}', filename)


def sparse_pgd_attack(run_dir, dataset, attack_name, basemodel_name, alpha, data, trainlog_path, ptb_rate=0.05, device='cpu'):
    print('Device', device)

    # Instantiating model
    from model import GCN, GAT, OurGCN, OurGAT
    if attack_name == 'attack_base':
        if basemodel_name == 'GCN':
            victim_model = GCN(dataset.num_features, 16,
                               dataset.num_classes).to(device)
        else:
            victim_model = GAT(dataset.num_features, 8,
                               dataset.num_classes).to(device)
        checkpoint_step = 0
    elif attack_name == 'attack_ours':
        if basemodel_name == 'GCN':
            victim_model = OurGCN(dataset.num_features, 16,
                                  dataset.num_classes, alpha=alpha).to(device)
        else:
            victim_model = OurGAT(dataset.num_features, 8,
                                  dataset.num_classes, alpha=alpha).to(device)
        checkpoint_step = 5
    victim_model = victim_model.to(device)
    print('Victim model', victim_model)

    # Loading checkpoint
    checkpoint_path = run_dir / ('model_'+str(checkpoint_step)+'.pt')
    checkpoint = torch.load(checkpoint_path)
    victim_model.load_state_dict(checkpoint['model'])
    print('Loaded checkpoint:', checkpoint_path)

    from trainer import Trainer
    # Testing with A
    ori_data = dataset[0].to(device)
    ori_adj = to_dense_adj(ori_data.edge_index, edge_attr=ori_data.edge_attr,
                           max_num_nodes=ori_data.num_nodes)[0]
    ori_trainer = Trainer(victim_model, ori_data, device, trainlog_path)
    (ori_train_acc, ori_val_acc, ori_test_acc), ori_logit = ori_trainer.test()
    print('Original Adjacency\n', ori_adj)
    print('GNN(X, A)\n', ori_train_acc, ori_val_acc, ori_test_acc)
    print(ori_logit)

    # Testing with A'
    aug_data = dataset[0].to(device)
    aug_data.edge_index = checkpoint['edge_index']
    aug_data.edge_attr = checkpoint['edge_attr']
    aug_adj = to_dense_adj(aug_data.edge_index, edge_attr=aug_data.edge_attr,
                           max_num_nodes=aug_data.num_nodes)[0].to(device)
    aug_trainer = Trainer(victim_model, aug_data, device, trainlog_path)
    (aug_train_acc, aug_val_acc, aug_test_acc), aug_logit = aug_trainer.test()
    print('Diff sum (original adj vs augmented adj):',
          (ori_adj != aug_adj).sum().item())
    print('Original trainer', ori_trainer.edge_index.shape)
    print('Augmented trainer', aug_trainer.edge_index.shape)
    print('Augmented Adjacency\n', aug_adj)
    print("GNN(X, A')\n", aug_train_acc, aug_val_acc, aug_test_acc)
    print(aug_logit)

    # Attack parameters
    adj, features, labels = aug_adj, data.x, data.test_mask
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
    perturbations = int(ptb_rate * (adj.sum()//2))
    from deeprobust.graph.utils import to_scipy
    adj, features, labels = preprocess(
        to_scipy(adj), to_scipy(features), labels.long().cpu(), preprocess_adj=False, device=device)

    # Setup attack model
    attack_model = SparsePGDAttack(model=victim_model,
                                   nnodes=adj.shape[0],
                                   loss_type='CE',
                                   device=device)
    attack_model.attack(features, aug_data.edge_index, aug_data.edge_attr, labels,
                        idx_train, n_perturbations=perturbations)
    # attack_model.attack(features, adj, labels,
    #                     idx_train, n_perturbations=perturbations)
    modified_adj = attack_model.modified_adj
    print('Modified_adj\n', modified_adj)
    print('Diff sum', (modified_adj != adj).sum())
    diff_idx = (modified_adj != adj).nonzero(as_tuple=False)
    for i, (row, col) in enumerate(diff_idx):
        if i < 3:
            print('Different index: (', str(row.item())+',', str(col.item())+')', 'ori', adj[row][col].item(
            ), '->', modified_adj[row][col].item())

    modified_edge_index, modified_edge_attr = dense_to_sparse(modified_adj)
    return modified_edge_index, modified_edge_attr, modified_adj


def pgd_attack(run_dir, dataset, attack_name, basemodel_name, alpha, data, trainlog_path, ptb_rate=0.05, device='cpu', gradlog_path=None):
    print('Device', device)

    # Instantiating model
    from model import GCN, GAT, OurGCN, OurGAT
    if attack_name == 'attack_base':
        if basemodel_name == 'GCN':
            victim_model = GCN(dataset.num_features, 16,
                               dataset.num_classes).to(device)
            link_pred = GCN4ConvSIGIR
        else:
            victim_model = GAT(dataset.num_features, 8,
                               dataset.num_classes).to(device)
            link_pred = GAT4ConvSIGIR
        checkpoint_step = 0
    elif attack_name == 'attack_ours':
        if basemodel_name == 'GCN':
            victim_model = OurGCN(dataset.num_features, 16,
                                  dataset.num_classes, alpha=alpha).to(device)
            link_pred = GCN4ConvSIGIR
        else:
            victim_model = OurGAT(dataset.num_features, 8,
                                  dataset.num_classes, alpha=alpha).to(device)
            link_pred = GAT4ConvSIGIR
        checkpoint_step = 5
    victim_model = victim_model.to(device)
    print('Victim model', victim_model)

    # Loading checkpoint
    checkpoint_path = run_dir / ('model_'+str(checkpoint_step)+'.pt')
    checkpoint = torch.load(checkpoint_path)
    victim_model.load_state_dict(checkpoint['model'])
    print('Loaded checkpoint:', checkpoint_path)

    from trainer import Trainer
    # Testing with A
    ori_data = dataset[0].to(device)
    ori_adj = to_dense_adj(ori_data.edge_index, edge_attr=ori_data.edge_attr,
                           max_num_nodes=ori_data.num_nodes)[0]
    ori_trainer = Trainer(victim_model, ori_data, device, trainlog_path)
    (ori_train_acc, ori_val_acc, ori_test_acc), ori_logit = ori_trainer.test()
    print('Original Adjacency\n', ori_adj)
    print('GNN(X, A)\n', ori_train_acc, ori_val_acc, ori_test_acc)
    print(ori_logit)

    # Testing with A'
    aug_data = dataset[0].to(device)
    aug_data.edge_index = checkpoint['edge_index']
    aug_data.edge_attr = checkpoint['edge_attr']
    aug_adj = to_dense_adj(aug_data.edge_index, edge_attr=aug_data.edge_attr,
                           max_num_nodes=aug_data.num_nodes)[0].to(device)
    aug_trainer = Trainer(victim_model, aug_data, device, trainlog_path)
    (aug_train_acc, aug_val_acc, aug_test_acc), aug_logit = aug_trainer.test()
    print('Diff sum (original adj vs augmented adj):',
          (ori_adj != aug_adj).sum().item())
    print('Original trainer', ori_trainer.edge_index.shape)
    print('Augmented trainer', aug_trainer.edge_index.shape)
    print('Augmented Adjacency\n', aug_adj)
    print("GNN(X, A')\n", aug_train_acc, aug_val_acc, aug_test_acc)
    print(aug_logit)

    # Attack parameters
    adj, features, labels = aug_adj, data.x, data.test_mask
    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
    perturbations = int(ptb_rate * (adj.sum()//2))
    from deeprobust.graph.utils import to_scipy
    adj, features, labels = preprocess(
        to_scipy(adj), to_scipy(features), labels.long().cpu(), preprocess_adj=False, device=device)

    # # Setup victim model
    # from deeprobust.graph.defense import GCN as DRGCN
    # victim_model = DRGCN(nfeat=features.shape[1], nclass=labels.max().item()+1, nhid=16,
    #                      dropout=0.5, weight_decay=5e-4, device=device)

    # victim_model = victim_model.to(device)
    # victim_model.fit(features, adj, labels, idx_train)

    # Setup attack model
    attack_model = PGDAttack(model=victim_model,
                             nnodes=adj.shape[0],
                             loss_type='CE',
                             device=device)
    attack_model.geometric_attack(features, adj, labels,
                                  idx_train, perturbations, aug_trainer, link_pred=link_pred, gradlog_path=gradlog_path)
    # attack_model.attack(features, adj, labels,
    #                     idx_train, n_perturbations=perturbations)
    modified_adj = attack_model.modified_adj
    print('Modified_adj\n', modified_adj)
    print('Diff sum', (modified_adj != adj).sum())
    diff_idx = (modified_adj != adj).nonzero(as_tuple=False)
    for i, (row, col) in enumerate(diff_idx):
        if i < 3:
            print('Different index: (', str(row.item())+',', str(col.item())+')', 'ori', adj[row][col].item(
            ), '->', modified_adj[row][col].item())

    modified_edge_index, modified_edge_attr = dense_to_sparse(modified_adj)
    return modified_edge_index, modified_edge_attr, modified_adj


def eval_metric(new_edge_index, gold_adj, node_degree, log_filename, fig_filename):
    new_edge_adj = to_dense_adj(
        new_edge_index, max_num_nodes=node_degree.size(0))[0]
    new_edge_adj[new_edge_adj > 1] = 1

    perf_stat = compare_topology(
        new_edge_adj, gold_adj, log_filename, fig_filename, add_loop=False)
    ppv = perf_stat['ppv']

    counter = [0 for _ in range(node_degree.size(0))]

    node_degree_list = node_degree.tolist()
    new_edge_index_list = new_edge_index.T.tolist()
    for i, j in new_edge_index_list:
        counter[i] += node_degree_list[i]
        counter[j] += node_degree_list[j]
    length = 0
    for degree in counter:
        if degree > 0:
            length += 1
    mean_degree = sum(counter)/length

    ppv = ppv / 100
    mean_degree = mean_degree
    metric = ppv

    ppv = round(ppv, 4)
    mean_degree = round(mean_degree, 4)
    metric = round(metric, 4)

    superprint(
        f'Metric: {metric} New Edge Precision: {ppv} Mean Node Degree: {mean_degree}', log_filename=log_filename)
    return metric


def log_run_metric(metric, test_accs, filename):
    superprint(
        f'Metric: {metric}', filename, overwrite=True)
    superprint(
        f'Test: {test_accs}', filename)
