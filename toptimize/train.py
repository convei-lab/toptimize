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
    cold_start,
    superprint
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

# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_vals, base_tests = [], []
our_vals, our_tests = [], []
noen_our_vals, noen_our_tests = [], []

cur_dir = Path(__file__).resolve().parent
exp_name = exp_alias + '_' + dataset_name + '_' + basemodel_name
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
    trainlog_path = run_dir / 'train_log.txt'
    step_perf_path = run_dir / 'step_perf.txt'
    run_perf_path = exp_dir / 'run_perf.txt'

    # Dataset
    dataset, data = load_data(dataset_path, dataset_name, device, use_gdc)
    data.edge_index = cold_start(data.edge_index, ratio=cold_start_ratio)
    log_dataset_stat(dataset, datastat_path)
    label = data.y
    one_hot_label = F.one_hot(data.y).float()
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
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
            model.parameters(), lr=0.01, weight_decay=5e-4)
    log_model_architecture(step, model, optimizer, archi_path, overwrite=True)

    trainer = Trainer(model, data, device,
                      trainlog_path, optimizer=optimizer)
    if basemodel_name == 'GCN':
        train_acc, val_acc, test_acc = trainer.fit(
            step, 200, lambda1, lambda2, use_last_epoch=False, use_loss_epoch=False)
    else:
        train_acc, val_acc, test_acc = trainer.fit(
            step, 300, lambda1, lambda2, use_last_epoch=False, use_loss_epoch=False)
    base_vals.append(val_acc)
    base_tests.append(test_acc)

    final, logit = trainer.infer()

    if eval_topo:
        perf_stat = evaluate_experiment(
            step, final, label, adj, gold_adj, confmat_dir, topofig_dir, tsne_dir)

    trainer.save_model(run_dir / ('model_'+str(step)+'.pt'), data)

    ##################################################
    ############## Training Our Model ################
    ##################################################

    step_vals, step_tests = [], []
    step_noen_vals, step_noen_tests = [], []
    wnb_group_name = exp_alias + '_run' + \
        str(run) + '_' + wandb.util.generate_id()

    for step in range(1, total_step + 1):
        teacher = trainer.checkpoint['logit']
        if eval_topo:
            prev_stat = perf_stat

        wnb_run = None
        if use_wnb:
            wnb_run = wandb.init(
                project="toptimize", name='Step'+str(step), group=wnb_group_name, config=args)
            wnb_run.watch(model, log='all')

        if basemodel_name == 'GCN':
            model = OurGCN(dataset.num_features, 16,
                           dataset.num_classes, alpha=alpha, beta=beta).to(device)
            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=5e-4),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=0.01)
            link_pred = GCN4ConvSIGIR
        else:
            model = OurGAT(dataset.num_features, 8,
                           dataset.num_classes, alpha=alpha, beta=beta).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.01, weight_decay=5e-4)
            link_pred = GAT4ConvSIGIR
        log_model_architecture(step, model, optimizer, archi_path)

        trainer = Trainer(model, data, device,
                          trainlog_path, optimizer)
        train_acc, val_acc, test_acc = trainer.fit(
            step, total_epoch, lambda1, lambda2, link_pred=link_pred, teacher=teacher, use_last_epoch=use_last_epoch, use_loss_epoch=use_loss_epoch, wnb_run=wnb_run)

        step_noen_vals.append(val_acc)
        step_noen_tests.append(test_acc)
        superprint(
            f'Non Ensembled Train {train_acc} Val {val_acc} Test {test_acc}', trainlog_path)

        # TODO check if logit in test func is identical to the infer's
        final, logit = trainer.infer()
        data.edge_index, data.edge_attr, adj = trainer.augment_topology(
            drop_edge=drop_edge)

        if eval_topo:
            perf_stat = evaluate_experiment(
                step, final, label, adj, gold_adj, confmat_dir, topofig_dir, tsne_dir, prev_stat)

        trainer.save_model(run_dir / ('model_'+str(step)+'.pt'), data)
        train_acc, val_acc, test_acc = trainer.ensemble(run_dir)

        superprint(
            f'Ensembled Train {train_acc} Val {val_acc} Test {test_acc}', trainlog_path)

        step_vals.append(val_acc)
        step_tests.append(test_acc)

        if use_wnb:
            if eval_topo:
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
    noen_our_vals.append(step_noen_vals[-1])
    noen_our_tests.append(step_noen_tests[-1])

    log_step_perf(step_vals, step_tests, step_noen_vals,
                  step_noen_tests, step_perf_path)

    run_dir = exp_dir / ('run_' + str(run))
    for file in run_dir.iterdir():
        if file.suffix == '.pt':
            file.unlink()

log_run_perf(base_vals, base_tests, our_vals, our_tests,
             noen_our_vals, noen_our_tests, run_perf_path)
