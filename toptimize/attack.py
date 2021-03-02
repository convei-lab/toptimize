
import argparse
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
    superprint,
    eval_metric,
    log_run_metric,
    pgd_attack,
    random_attack,
    compare_topology,
)
from trainer import Trainer
from model import GCN, GAT, OurGCN, OurGAT
from utils import evaluate_experiment

parser = argparse.ArgumentParser()
parser.add_argument('att_alias', type=str,
                    help='Experiment alias')
parser.add_argument('attack_type', type=str, choices=[
                    'pgd_attack', 'random_attack'])
parser.add_argument('victim_name', type=str,
                    help='expalias_dataset_basemodel')
parser.add_argument('-vm', '--victim_model_step', type=int, default=20,
                    help='The model checkpoint of victim model')
parser.add_argument('-vt', '--victim_topo_step', type=int, default=20,
                    help='The topology checkpoint of victim model')
parser.add_argument('-b', '--basemodel', default='GCN',
                    type=str, choices=['GCN', 'GAT'])
parser.add_argument('-tr', '--total_run', type=int,
                    default=20, help='Starting from run 0, specify the last run')
parser.add_argument('-ts', '--total_step', default=5, type=int)
parser.add_argument('-te', '--total_epoch', default=300, type=int)
parser.add_argument('-s', '--seed', default=None,
                    type=int, help='If none, random seed.')
parser.add_argument('-hs', '--hidden_sizes', default=None, type=int)
parser.add_argument('-l1', '--lambda1', default=1, type=float)
parser.add_argument('-l2', '--lambda2', default=10, type=float)
parser.add_argument('-t', '--tau', default=10, type=float)
parser.add_argument('-n', '--beta', default=-3, type=float)
parser.add_argument('-csr', '--cold_start_ratio', default=1.0, type=float)
parser.add_argument('-ptb', '--ptb_rate', default=0.05, type=float)
parser.add_argument('-ca', '--compare_attacked', action='store_true')
parser.add_argument('-et', '--eval_topo', action='store_true',
                    help='Save confusion matrix, TSNE, and visualize topology.')
parser.add_argument('-le', '--use_last_epoch', action='store_true',
                    help='Use last epoch as the final epoch')
parser.add_argument('-o', '--use_loss_epoch', action='store_true',
                    help='Use best loss epoch as the final epoch')
parser.add_argument('-de', '--drop_edge', action='store_true',
                    help='Remove edge, not just adding')
parser.add_argument('-wnb', '--use_wnb', action='store_true',
                    help='Use weights and biases for visual logging')
parser.add_argument('-gdc', '--use_gdc', action='store_true',
                    help='Use GDC preprocessing for GCN.')
parser.add_argument('-z', '--use_metric', action='store_true')
parser.add_argument('-sm', '--save_model', action='store_true')
parser.add_argument('-ea', '--eval_new_adj', action='store_true')
args = parser.parse_args()

args.seed = args.seed if args.seed else random.randint(0, 2**32 - 1)
att_alias = args.att_alias
victim_name = args.victim_name
victim_alias, dataset_name, vic_basemodel_name = victim_name.split('_')
total_run = args.total_run
victim_model_step = args.victim_model_step
victim_topo_step = args.victim_topo_step
args.vic_basemodel_name = vic_basemodel_name
attack_type = args.attack_type
basemodel_name = args.basemodel
args.dataset_name = dataset_name
total_step = args.total_step
total_epoch = args.total_epoch
seed = args.seed
hidden_sizes = args.hidden_sizes
lambda1 = args.lambda1
lambda2 = args.lambda2
alpha = args.tau
beta = args.beta
eval_topo = args.eval_topo
cold_start_ratio = args.cold_start_ratio
use_last_epoch = args.use_last_epoch
use_loss_epoch = args.use_loss_epoch
use_wnb = args.use_wnb
drop_edge = args.drop_edge
use_gdc = args.use_gdc
use_metric = args.use_metric
save_model = args.save_model
ptb_rate = args.ptb_rate
compare_attacked = args.compare_attacked
eval_new_adj = args.eval_new_adj

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cur_dir = Path(__file__).resolve().parent
victim_dir = (cur_dir.parent / 'experiment' / victim_name).resolve()

attack_name = f"{att_alias}-{victim_name}-r{total_run}-m{victim_model_step}-t{victim_topo_step}"
attack_dir = (cur_dir.parent / 'experiment' / attack_type /
              victim_name / attack_name).resolve()
attack_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
safe_remove_dir(attack_dir)

base_vals, base_tests = [], []
our_vals, our_tests = [], []
noen_our_vals, noen_our_tests = [], []
if use_metric:
    all_run_metric = []


for run in list(range(total_run)):
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@ RUN',
          run, ' @@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    # Directories
    run_name = 'run_' + str(run)
    run_dir = victim_dir / ('run_' + str(run))
    attack_run_dir = attack_dir / ('run_' + str(run))
    confmat_dir = attack_run_dir / 'confmat'
    topofig_dir = attack_run_dir / 'topofig'
    tsne_dir = attack_run_dir / 'tsne'
    metric_dir = run_dir / 'metric'
    attack_run_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    confmat_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    topofig_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    tsne_dir.mkdir(mode=0o777, parents=True, exist_ok=True)
    metric_dir.mkdir(mode=0o777, parents=True, exist_ok=True)

    # Path
    dataset_path = (Path(__file__) /
                    '../../data').resolve() / dataset_name
    hyper_path = attack_dir / 'hyper.txt'
    datastat_path = attack_run_dir / 'data_stat.txt'
    archi_path = attack_dir / 'model_archi.txt'
    attacklog_path = attack_run_dir / 'attack_log.txt'
    trainlog_path = attack_run_dir / 'train_log.txt'
    step_perf_path = attack_run_dir / 'step_perf.txt'
    run_perf_path = attack_dir / 'run_perf.txt'
    metric_fig_path = metric_dir / 'metric.png'
    metric_txt_path = metric_dir / 'metric.txt'
    vic_mod_ckpt_path = run_dir / f'model_{victim_model_step}.pt'
    vic_topo_ckpt_path = run_dir / f'model_{victim_topo_step}.pt'
    victim_ckpt_path = (vic_mod_ckpt_path, vic_topo_ckpt_path)
    log_hyperparameters(args, hyper_path)

    # Dataset
    dataset, data = load_data(
        dataset_path, dataset_name, device, use_gdc)
    orig_adj = to_dense_adj(data.edge_index, edge_attr=data.edge_attr)
    node_degree = orig_adj.sum(dim=1)[0]
    ################## Attack Model ###################
    data.edge_index = cold_start(data.edge_index, ratio=cold_start_ratio)
    if attack_type == 'pgd_attack':
        data.edge_index, data.edge_attr = pgd_attack(
            dataset, vic_basemodel_name, victim_ckpt_path, attacklog_path, ptb_rate=ptb_rate, device=device, compare_attacked=compare_attacked)
    elif attack_type == 'random_attack':
        data.edge_index, data.edge_attr = random_attack(
            dataset, vic_basemodel_name, victim_ckpt_path, attacklog_path, ptb_rate=ptb_rate, device=device, compare_attacked=compare_attacked)
    ###############  End of Attack Model ##############
    label = data.y
    one_hot_label = F.one_hot(data.y).float()
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
    gold_adj = torch.matmul(one_hot_label, one_hot_label.T)
    log_dataset_stat(data, dataset, datastat_path)
    ##################################################
    ############## Training Base Model ###############
    ##################################################

    step = 0
    # basemodel_name = 'GAT'
    if basemodel_name == 'GCN':
        hidden_sizes = hidden_sizes if hidden_sizes else 16
        model = GCN(dataset.num_features, hidden_sizes,
                    dataset.num_classes).to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.01)
    else:
        hidden_sizes = hidden_sizes if hidden_sizes else 8
        model = GAT(dataset.num_features, hidden_sizes,
                    dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.005, weight_decay=0.0005)
    log_model_architecture(step, model, optimizer,
                           archi_path, overwrite=True)

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

    trainer.save_model(attack_run_dir / ('model_'+str(step)+'.pt'), data)

    if eval_topo:
        final, logit = trainer.infer()
        perf_stat = evaluate_experiment(
            step, final, label, adj, gold_adj, confmat_dir, topofig_dir, tsne_dir)

        # test_adj = torch.zeros_like(adj)
        # test_adj[data.test_mask, :] = adj[data.test_mask, :]
        # test_adj = test_adj + test_adj.T
        # test_adj[test_adj > 1] = 1
        # perf_stat = evaluate_experiment(
        #     step, final, label, test_adj, gold_adj, confmat_dir, topofig_dir, tsne_dir)

        # train_adj = torch.zeros_like(adj)
        # train_adj[data.train_mask, :] = adj[data.train_mask, :]
        # train_adj = train_adj + train_adj.T
        # train_adj[train_adj > 1] = 1
        # perf_stat = evaluate_experiment(
        #     step, final, label, train_adj, gold_adj, confmat_dir, topofig_dir, tsne_dir)

    ##################################################
    ############## Training Our Model ################
    ##################################################

    step_vals, step_tests = [], []
    step_noen_vals, step_noen_tests = [], []
    wnb_group_name = att_alias + '_run' + \
        str(run) + '_' + wandb.util.generate_id()

    if use_metric:
        all_step_new_edge = None

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
            hidden_sizes = hidden_sizes if hidden_sizes else 16
            model = OurGCN(dataset.num_features, hidden_sizes,
                           dataset.num_classes, alpha=alpha, beta=beta).to(device)
            optimizer = torch.optim.Adam([
                dict(params=model.conv1.parameters(), weight_decay=5e-4),
                dict(params=model.conv2.parameters(), weight_decay=0)
            ], lr=0.01)
            link_pred = GCN4ConvSIGIR
        else:
            hidden_sizes = hidden_sizes if hidden_sizes else 8
            model = OurGAT(dataset.num_features, hidden_sizes,
                           dataset.num_classes, alpha=alpha, beta=beta).to(device)
            print('Our model model.parameters()', model.parameters())
            input()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=0.005, weight_decay=5e-4)
            link_pred = GAT4ConvSIGIR
        log_model_architecture(step, model, optimizer, archi_path)

        trainer = Trainer(model, data, device,
                          trainlog_path, optimizer)
        noen_train_acc, noen_val_acc, noen_test_acc = trainer.fit(
            step, total_epoch, lambda1, lambda2, link_pred=link_pred, teacher=teacher, use_last_epoch=use_last_epoch, use_loss_epoch=use_loss_epoch, wnb_run=wnb_run)

        step_noen_vals.append(noen_val_acc)
        step_noen_tests.append(noen_test_acc)
        superprint(
            f'Non Ensembled Train {noen_train_acc} Val {noen_val_acc} Test {noen_test_acc}', trainlog_path)

        data.edge_index, data.edge_attr, adj, new_edge, new_adj = trainer.augment_topology(
            drop_edge=drop_edge)
        if use_metric:
            all_step_new_edge = new_edge.clone().detach() if all_step_new_edge is None else torch.cat([
                all_step_new_edge, new_edge], dim=1)

        trainer.save_model(
            attack_run_dir / ('model_'+str(step)+'.pt'), data)
        train_acc, val_acc, test_acc = trainer.ensemble(attack_run_dir)

        if eval_new_adj:
            compare_topology(new_adj, gold_adj, trainlog_path,
                             add_loop=False, reset_log=False)

        if eval_topo:
            perf_stat = evaluate_experiment(
                step, final, label, adj, gold_adj, confmat_dir, topofig_dir, tsne_dir, prev_stat)
        superprint(
            f'\nRun {run} Ensembled Train {train_acc} Val {val_acc} Test {test_acc}', trainlog_path)

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

    if total_step > 0:
        our_vals.append(val_acc)
        our_tests.append(test_acc)
        noen_our_vals.append(noen_val_acc)
        noen_our_tests.append(noen_test_acc)

        log_step_perf(step_vals, step_tests, step_noen_vals,
                      step_noen_tests, step_perf_path)

        if use_metric:
            if all_step_new_edge is not None:
                print('all_step_new_edge', all_step_new_edge,
                      all_step_new_edge.shape)
                if all_step_new_edge.nelement() != 0:
                    metric = eval_metric(all_step_new_edge, gold_adj, node_degree,
                                         metric_txt_path, metric_fig_path)
                else:
                    metric = -1
                all_run_metric.append(metric)
            print('all_run_metric', all_run_metric, len(all_run_metric))

    if not save_model:
        run_dir = attack_dir / ('run_' + str(run))
        for file in run_dir.iterdir():
            if file.suffix == '.pt':
                file.unlink()

    log_run_perf(base_vals, base_tests, our_vals, our_tests,
                 run_perf_path, noen_our_vals, noen_our_tests)
if use_metric:
    log_run_metric(all_run_metric, test_acc, filename=metric_txt_path)
