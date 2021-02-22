import argparse
from numpy.lib.function_base import append
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCN4ConvSIGIR, GAT4ConvSIGIR
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
)
from trainer import Trainer
from model import GCN, GAT, OurGCN, OurGAT
from utils import evaluate_experiment

parser = argparse.ArgumentParser()
parser.add_argument('exp_alias', type=str)
parser.add_argument('-b', '--basemodel', default='GCN', type=str)
parser.add_argument('-d', '--dataset', default='Cora', type=str)
parser.add_argument('-r', '--total_run', default=2, type=int)
parser.add_argument('-t', '--total_step', default=5, type=int)
parser.add_argument('-e', '--total_epoch', default=300, type=int)
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-l1', '--lambda1', default=1, type=float)
parser.add_argument('-l2', '--lambda2', default=10, type=float)
parser.add_argument('-l', '--use_last', action='store_true')
parser.add_argument('-dr', '--drop_edge', action='store_true')
parser.add_argument('-w', '--use_wnb', action='store_true')
parser.add_argument('-g', '--use_gdc', action='store_true',
                    help='Use GDC preprocessing for GCN.')
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
use_last_epoch = args.use_last
use_wnb = args.use_wnb
drop_edge = args.drop_edge
use_gdc = args.use_gdc

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_vals, base_tests = [], []
our_vals, our_tests = [], []

exp_name = exp_alias + '_' + dataset_name + '_' + basemodel_name
cur_dir = Path(__file__).resolve().parent
exp_dir = (cur_dir.parent / 'experiment' / exp_name).resolve()
safe_remove_dir(exp_dir)

for run in list(range(total_run)):

    # Directories
    run_name = 'run_' + str(run)
    run_dir = exp_dir / ('run_' + str(run))
    confmat_dir = run_dir / 'confmat'
    topofig_dir = run_dir / 'topofig'
    tsne_dir = run_dir / 'tsne'
    confmat_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    topofig_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    tsne_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    # Path
    dataset_path = (Path(__file__) / '../../data').resolve() / dataset_name
    hyper_path = exp_dir / 'hyper.txt'
    datastat_path = exp_dir / 'data_stat.txt'
    archi_path = exp_dir / 'model_archi.txt'
    basemodel_path = run_dir / 'base_model.pt'
    ourmodel_path = run_dir / 'our_model.pt'
    trainlog_path = run_dir / 'train_log.txt'
    step_perf_path = run_dir / 'step_perf.txt'
    run_perf_path = exp_dir / 'run_perf.txt'

    # Dataset
    dataset, data = load_data(dataset_path, dataset_name, device, use_gdc)
    log_dataset_stat(dataset, datastat_path)
    label = data.y
    one_hot_label = F.one_hot(data.y).float()
    adj = to_dense_adj(data.edge_index)[0]
    gold_adj = torch.matmul(one_hot_label, one_hot_label.T)

    log_hyperparameters(args, hyper_path)

    ###################################################
    ############## Training Base Model ################
    ###################################################

    step = 0

    if basemodel_name == 'GCN':
        model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.01)
    else:
        model = GAT(dataset.num_features, 8, dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.005, weight_decay=5e-4)
    log_model_architecture(step, model, optimizer, archi_path, overwrite=True)

    trainer = Trainer(model, data, device,
                      trainlog_path, use_last_epoch, optimizer=optimizer)
    train_acc, val_acc, test_acc = trainer.train(
        step, total_epoch, lambda1, lambda2)
    base_vals.append(val_acc)
    base_tests.append(test_acc)

    final, logit = trainer.infer()

    perf_stat = evaluate_experiment(
        step, final, label, adj, gold_adj, confmat_dir, topofig_dir, tsne_dir)

    trainer.save_model(basemodel_path, data)

    ##################################################
    ############## Training Our Model ################
    ##################################################

    step_vals, step_tests = [], []
    wnb_group_name = exp_alias + '_run' + \
        str(run) + '_' + wandb.util.generate_id()

    alpha, omega = 10, -3
    for step in range(1, total_step + 1):
        teacher = trainer.checkpoint['logit']
        prev_stat = perf_stat

        wnb_run = None
        if use_wnb:
            wnb_run = wandb.init(
                project="toptimize", name='Step'+str(step), group=wnb_group_name, config=args)
            wnb_run.watch(model, log='all')

        if basemodel_name == 'GCN':
            model = OurGCN(dataset.num_features, 16,
                           dataset.num_classes, alpha=alpha, omega=omega).to(device)
            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=5e-4),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=0.01)
            link_pred = GCN4ConvSIGIR
        else:
            model = OurGAT(dataset.num_features, 8,
                           dataset.num_classes, alpha=alpha, omega=omega).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.005, weight_decay=5e-4)
            link_pred = GAT4ConvSIGIR
        log_model_architecture(step, model, optimizer, archi_path)

        trainer = Trainer(model, data, device,
                          trainlog_path, use_last_epoch, optimizer)
        train_acc, val_acc, test_acc = trainer.train(
            step, total_epoch, lambda1, lambda2, link_pred=link_pred, teacher=teacher, wnb_run=wnb_run)

        step_vals.append(val_acc)
        step_tests.append(test_acc)

        final, logit = trainer.infer()
        data.edge_index, data.edge_attr, adj = trainer.augment_topology(
            drop_edge=drop_edge)

        perf_stat = evaluate_experiment(
            step, final, label, adj, gold_adj, confmat_dir, topofig_dir, tsne_dir, prev_stat)

        trainer.save_model(ourmodel_path, trainer)

        if use_wnb:
            wnb_run.log(perf_stat)
            for key, val in perf_stat.items():
                wandb.run.summary[key] = val
            wandb.run.summary['train_acc'] = train_acc
            wandb.run.summary['val_acc'] = val_acc
            wandb.run.summary['test_acc'] = test_acc
            wandb.finish()

        print()

    our_vals.append(val_acc)
    our_tests.append(test_acc)

    log_step_perf(step_vals, step_tests, step_perf_path)

log_run_perf(base_vals, base_tests, our_vals, our_tests, run_perf_path)
