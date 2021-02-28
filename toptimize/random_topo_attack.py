import argparse
from numpy.lib.function_base import append
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN4ConvSIGIR, GAT4ConvSIGIR
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils import to_dense_adj
import wandb
import random
import numpy as np
from pathlib import Path
from utils import (
    safe_remove_dir,
    load_data,
    log_dataset_stat,
    log_model_architecture,
    log_step_perf,
    log_run_perf,
    log_hyperparameters,
    sparse_pgd_attack,
    pgd_attack,
    superprint
)
from trainer import Trainer
from model import GCN, GAT, OurGCN, OurGAT
from utils import evaluate_experiment, add_random_edge

parser = argparse.ArgumentParser()
parser.add_argument('exp_alias', type=str)
parser.add_argument('-b', '--basemodel', default='GCN', type=str)
parser.add_argument('-d', '--dataset', default='Cora', type=str)
parser.add_argument('-r', '--total_run', default=2, type=int)
parser.add_argument('-t', '--total_step', default=5, type=int)
parser.add_argument('-e', '--total_epoch', default=300, type=int)
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-l', '--lambda1', default=1, type=float)
parser.add_argument('-k', '--lambda2', default=10, type=float)
parser.add_argument('-m', '--alpha', default=10, type=float)
parser.add_argument('-n', '--beta', default=-3, type=float)
parser.add_argument('-c', '--cold_start_ratio', default=1.0, type=float)
parser.add_argument('-x', '--eval_topo', action='store_true')
parser.add_argument('-a', '--use_last_epoch', action='store_true')
parser.add_argument('-o', '--use_loss_epoch', action='store_true')
parser.add_argument('-p', '--drop_edge', action='store_true')
parser.add_argument('-w', '--use_wnb', action='store_true')
parser.add_argument('-g', '--use_gdc', action='store_true',
                    help='Use GDC preprocessing for GCN.')
parser.add_argument('-i', '--ptb_rate', default=0.05, type=float)
parser.add_argument('-rer', '--random_edge_ratio', default=0.05, type=float)
args = parser.parse_args()

exp_alias = args.exp_alias
dataset_name = args.dataset
basemodel_name = args.basemodel
total_run = args.total_run
total_step = args.total_step
total_epoch = args.total_epoch
seed = args.seed
lambda1 = args.lambda1
lambda2 = args.lambda2
alpha = args.alpha
beta = args.beta
eval_topo = args.eval_topo
cold_start_ratio = args.cold_start_ratio
use_last_epoch = args.use_last_epoch
use_loss_epoch = args.use_loss_epoch
use_wnb = args.use_wnb
drop_edge = args.drop_edge
use_gdc = args.use_gdc
ptb_rate = args.ptb_rate
ratio = args.random_edge_ratio

# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_vals, base_tests = [], []
our_vals, our_tests = [], []

cur_dir = Path(__file__).resolve().parent
exp_name = exp_alias + '_' + dataset_name + '_' + basemodel_name
exp_dir = (cur_dir.parent / 'experiment' / exp_name).resolve()


attack_dir = (cur_dir.parent / 'experiment/random_attack' /
                  exp_name).resolve()
attack_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
safe_remove_dir(attack_dir)

for run in list(range(total_run)):
    # Directories
    run_name = 'run_' + str(run)
    run_dir = exp_dir / ('run_' + str(run))
    attack_run_dir = attack_dir / ('run_' + str(run))
    confmat_dir = attack_run_dir / 'confmat'
    topofig_dir = attack_run_dir / 'topofig'
    tsne_dir = attack_run_dir / 'tsne'
    attack_run_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    confmat_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    topofig_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    tsne_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    # Path
    dataset_path = (Path(__file__) / '../../data').resolve() / dataset_name
    hyper_path = attack_dir / 'hyper.txt'
    datastat_ori_path = attack_run_dir / 'data_stat_ori.txt' 
    datastat_per_path = attack_run_dir / 'data_stat_per.txt'
    archi_path = attack_dir / 'model_archi.txt'
    trainlog_ori_path = attack_run_dir / 'train_ori_log.txt'
    trainlog_per_path = attack_run_dir / 'train_per_log.txt'
    step_perf_path = attack_run_dir / 'step_perf.txt'
    run_perf_path = attack_dir / 'run_perf.txt'
    gradlog_path = attack_run_dir / 'grad_fig.pdf'

    # Dataset
    dataset, data = load_data(dataset_path, dataset_name, device, use_gdc)
    # log_dataset_stat(data, dataset, datastat_path)
    label = data.y
    one_hot_label = F.one_hot(data.y).float()
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    gold_adj = torch.matmul(one_hot_label, one_hot_label.T)

    # log_hyperparameters(args, hyper_path)

    ###################################################
    ################## Attack Model ###################
    ###################################################

    # Instantiating model
    from model import GCN, GAT, OurGCN, OurGAT
    if basemodel_name == 'GCN':
        model = GCN(dataset.num_features, 16,
                            dataset.num_classes, cached=False).to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.01)
    else:
        model = GAT(dataset.num_features, 8,
                            dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.005, weight_decay=0.0005)
    model.conv1._cached_edge_index = None
    model.conv2._cached_edge_index = None
    # Loading checkpoint
    checkpoint_step = 0
    checkpoint_path = run_dir / ('model_'+str(checkpoint_step)+'.pt')
    checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['model'])
    print('Loaded checkpoint:', checkpoint_path)

    from trainer import Trainer
    # Testing with A
    ori_data = dataset[0].to(device)
    ori_data.edge_index, ori_data.edge_attr = add_random_edge(ori_data.edge_index, ori_data.num_nodes, ratio=ratio)
    ori_trainer = Trainer(model, ori_data, device, trainlog_ori_path, optimizer=optimizer)

    if basemodel_name == 'GCN':
        train_acc, val_acc, test_acc = ori_trainer.fit(
            0, 200, lambda1, lambda2, use_last_epoch=False, use_loss_epoch=False)
    else:
        train_acc, val_acc, test_acc = ori_trainer.fit(
            0, 300, lambda1, lambda2, use_last_epoch=False, use_loss_epoch=False)

    (ori_train_acc, ori_val_acc, ori_test_acc), ori_logit = ori_trainer.test()

    base_tests.append(ori_test_acc)
    base_vals.append(ori_val_acc)

    if basemodel_name == 'GCN':
        model = GCN(dataset.num_features, 16,
                            dataset.num_classes, cached=False).to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.01)
    else:
        model = GAT(dataset.num_features, 8,
                            dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.005, weight_decay=0.0005)
    model.conv1._cached_edge_index = None
    model.conv2._cached_edge_index = None

    # Loading checkpoint
    checkpoint_step = 10
    checkpoint_path = run_dir / ('model_'+str(checkpoint_step)+'.pt')
    checkpoint = torch.load(checkpoint_path)
    print('Loaded checkpoint:', checkpoint_path)

    # Testing with A'
    aug_data = dataset[0].to(device)
    aug_data.edge_index = checkpoint['edge_index']
    aug_data.edge_attr = checkpoint['edge_attr']
    aug_data.edge_index, aug_data.edge_attr = add_random_edge(aug_data.edge_index, aug_data.num_nodes, ratio=ratio)

    aug_trainer = Trainer(model, aug_data, device, trainlog_per_path, optimizer=optimizer)

    if basemodel_name == 'GCN':
        train_acc, val_acc, test_acc = aug_trainer.fit(
            0, 200, lambda1, lambda2, use_last_epoch=False, use_loss_epoch=False)
    else:
        train_acc, val_acc, test_acc = aug_trainer.fit(
            0, 300, lambda1, lambda2, use_last_epoch=False, use_loss_epoch=False)

    (aug_train_acc, aug_val_acc, aug_test_acc), aug_logit = aug_trainer.test()
    our_tests.append(aug_test_acc)
    our_vals.append(aug_val_acc)

    print('Original logit', ori_logit)
    print('Aug_logit', aug_logit)
    print('Original trainer', ori_trainer.edge_index.shape)
    print('Augmented trainer', aug_trainer.edge_index.shape)
    print('Difference in edges', aug_trainer.edge_index.size(1)-ori_trainer.edge_index.size(1))
    print('GNN(X, A)\n', ori_train_acc, ori_val_acc, ori_test_acc)
    print("GNN(X, A')\n", aug_train_acc, aug_val_acc, aug_test_acc)


log_run_perf(base_vals, base_tests, our_vals, our_tests, run_perf_path)