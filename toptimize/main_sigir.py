import os.path as osp
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GCN4ConvSIGIR, GATConv, GAT4ConvSIGIR  # noqa
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.sparse import dense_to_sparse
from copy import deepcopy
import wandb
import random
import numpy as np
from pathlib import Path
from utils import (
    percentage, safe_remove_file,
    log_dataset_stat,
    log_label_relation,
    log_model_architecture,
    log_training,
    save_model,
    compare_topology,
    plot_tsne,
    plot_sorted_topology_with_gold_topology,
    superprint
)


parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()

method_name = 'GCNConv4SIGIR_woDL'
dataset_name = 'Cora'
basemodel = 'GCN'
runs = list(range(2))
total_z = 100
total_epoch = 1000

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for run in runs:

    # Path
    exp_name = method_name + '_' + dataset_name + \
        '_' + basemodel + '_' + str(run)
    exp_path = (Path.cwd() / '..' / 'experiment' / exp_name).resolve()
    exp_path.mkdir(mode=0o777, parents=True, exist_ok=True)

    # Experiment deliverables
    datastat_path = exp_path / 'datastat.txt'
    lblrel_path = exp_path / 'lblrel.txt'
    mdlarc_path = exp_path / 'mdlarc.txt'
    basemodel_path = exp_path / 'basemodel.pt'
    ourmodel_path = exp_path / 'ourmodel.pt'
    topology_path = exp_path / 'topology.txt'
    trainlog_path = exp_path / 'trainlog.txt'
    ressum_path = exp_path / 'ressum.txt'

    # Data path
    data_path = osp.join(osp.dirname(osp.realpath(__file__)),
                         '..', 'data', dataset_name)
    dataset = Planetoid(data_path, dataset_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    # Print Data Info.
    # input('Dataset Statistics')
    log_dataset_stat(dataset, data, datastat_path)
    # input('Label Relation')
    log_label_relation(data, lblrel_path)

    if args.use_gdc:
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
                    sparsification_kwargs=dict(method='topk', k=128,
                                               dim=0), exact=True)
        data = gdc(data)

    # print('data.edge_index', data.edge_index, data.edge_index.shape)
    # mask = torch.randint(0, 3 + 1, (1, data.edge_index.size(1)))[0]
    # print('mask', mask, mask.shape)
    # mask = mask >= 3
    # print('mask', mask, mask.shape)
    # data.edge_index = data.edge_index[:, mask]
    # print('data.edge_index', data.edge_index, data.edge_index.shape)

    ###################################################
    ############## Training Base Model ################
    ###################################################
    z = 0

    class GCN(torch.nn.Module):
        def __init__(self):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, 16, cached=True,
                                 normalize=not args.use_gdc)
            self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                                 normalize=not args.use_gdc)

        def forward(self):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
            final = self.conv2(x, edge_index, edge_weight)
            return final, F.log_softmax(final, dim=1)

    class GAT(torch.nn.Module):
        def __init__(self):
            super(GAT, self).__init__()
            self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
            # On the Pubmed dataset, use heads=8 in conv2.
            self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                                 dropout=0.6)

        def forward(self):
            x = F.dropout(data.x, p=0.6, training=self.training)
            x = F.elu(self.conv1(x, data.edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            final = self.conv2(x, data.edge_index)
            return final, F.log_softmax(final, dim=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None
    if basemodel == 'GCN':
        model = GCN().to(device)
    elif basemodel == 'GAT':
        model = GAT().to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=0.01)  # Only perform weight-decay on first convolution.

    # input('Model architecture')
    safe_remove_file(mdlarc_path)
    log_model_architecture(model, optimizer, mdlarc_path)

    def train():
        model.train()
        optimizer.zero_grad()
        _final, logits = model()

        task_loss = F.nll_loss(
            logits[data.train_mask], data.y[data.train_mask])
        # print('Task loss', task_loss)

        total_loss = task_loss

        total_loss.backward()
        optimizer.step()

    @torch.no_grad()
    def test():
        model.eval()
        (_final, logits), accs = model(), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            acc = percentage(acc)
            accs.append(acc)
        return accs

    @torch.no_grad()
    def distillation(best_model):
        best_model.eval()
        final, logits = best_model()
        pred = logits.max(1)[1]
        Y = F.one_hot(pred).float()
        YYT = torch.matmul(Y, Y.T)
        return final, logits, YYT

    # input('Start traning')
    safe_remove_file(trainlog_path)
    best_val_acc = test_acc = 0
    log_training(f'Start Training {z}', trainlog_path)
    log_training(f'*********************************', trainlog_path)
    for epoch in range(1, 201):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model)
            best_edge, best_weight = data.edge_index, data.edge_attr
            test_acc = tmp_test_acc
        log_text = f'\tEpoch: {epoch} Train: {train_acc} Val: {val_acc} Test: {test_acc}'
        log_training(log_text, trainlog_path)
    log_training(
        f'Z: {z} Val. Acc.: {best_val_acc} Test Acc.: {test_acc}', trainlog_path)
    log_training(f'Finished Training {z}', trainlog_path)
    log_training(f'*********************************', trainlog_path)

    final, logits, YYT = distillation(best_model)

    A = to_dense_adj(data.edge_index)[0]
    A.fill_diagonal_(1)
    gold_Y = F.one_hot(data.y).float()
    gold_A = torch.matmul(gold_Y, gold_Y.T)
    # sorted_gold_Y, sorted_Y_indices = torch.sort(data.y, descending=False)
    # sorted_gold_Y = F.one_hot(sorted_gold_Y).float()
    # sorted_gold_A = torch.matmul(sorted_gold_Y, sorted_gold_Y.T)

    # input('Compare toplogy')
    perf_stat = compare_topology(A, data, exp_path/('confmat'+str(z)+'.txt'),
                                 exp_path/('confmat'+str(z)+'.png'))

    # input('TSNE')
    plot_tsne(final, data.y, exp_path/('tsne_'+str(z)+'.png'))

    # plot_topology(A, data, 'A_original.png')
    # plot_topology(gold_A, data, 'A_gold.png')
    # plot_topology(sorted_gold_A, data, 'A_sorted_gold.png')
    # plot_topology(A, data, 'A_sorted_original.png', sorting=True)
    # plot_topology(gold_A, data, 'A_sorted_gold2.png', sorting=True)
    # plot_sorted_topology_with_gold_topology(A, gold_A, data, 'A_original_with_gold_'+str(z)+'.png', sorting=False)
    # input('Plot topology')
    plot_sorted_topology_with_gold_topology(
        A, gold_A, data, exp_path/('topofig'+str(z)+'.png'), sorting=True)

    save_model(best_model, final, best_edge, best_weight, basemodel_path)

    ##################################################
    ############## Training Our Model ################
    ##################################################

    val_accs, test_accs = [], []
    wnb_group_name = 'run'+str(run) + '_' + wandb.util.generate_id()

    for z in range(1, total_z + 1):
        prev_logits = logits
        prev_stat = perf_stat
        prev_tp, prev_fp = prev_stat['tp'], prev_stat['fp']

        wnb_run = wandb.init(project="toptimize",
                             name='z'+str(z), group=wnb_group_name)
        wandb.watch(model, log='all')

        class GCN(torch.nn.Module):
            def __init__(self):
                super(GCN, self).__init__()
                self.conv1 = GCN4ConvSIGIR(dataset.num_features, 16, cached=True,
                                           normalize=not args.use_gdc)
                self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                                     normalize=not args.use_gdc)

            def forward(self):
                x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
                x = F.relu(self.conv1(x, edge_index, edge_weight))
                x = F.dropout(x, training=self.training)
                final = self.conv2(x, edge_index, edge_weight)
                return final, F.log_softmax(final, dim=1)

        class GAT(torch.nn.Module):
            def __init__(self):
                super(GAT, self).__init__()
                self.conv1 = GAT4ConvSIGIR(
                    dataset.num_features, 8, heads=8, dropout=0.6)
                # On the Pubmed dataset, use heads=8 in conv2.
                self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                                     dropout=0.6)

            def forward(self):
                x = F.dropout(data.x, p=0.6, training=self.training)
                x = F.elu(self.conv1(x, data.edge_index))
                x = F.dropout(x, p=0.6, training=self.training)
                final = self.conv2(x, data.edge_index)
                return final, F.log_softmax(final, dim=1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = None
        if basemodel == 'GCN':
            model = GCN().to(device)
        elif basemodel == 'GAT':
            model = GAT().to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam([
            dict(params=model.conv1.parameters(), weight_decay=5e-4),
            dict(params=model.conv2.parameters(), weight_decay=0)
        ], lr=0.01)  # Only perform weight-decay on first convolution.

        # input('Model architecture')
        safe_remove_file(mdlarc_path)
        log_model_architecture(model, optimizer, mdlarc_path)

        def train():

            model.train()
            optimizer.zero_grad()
            _final, logits = model()

            task_loss = F.nll_loss(
                logits[data.train_mask], data.y[data.train_mask])
            # print('Task loss', task_loss)

            link_loss = 0
            if basemodel == 'GCN':
                link_loss = GCN4ConvSIGIR.get_link_prediction_loss(model)
            elif basemodel == 'GAT':
                link_loss = GAT4ConvSIGIR.get_link_prediction_loss(model)
            # print('Link loss', link_loss)

            # dist_loss = F.mse_loss(final, prev_logits, reduction = 'mean')
            dist_loss = F.kl_div(
                logits, prev_logits, reduction='none', log_target=True).mean()
            # dist_loss = torch.distributions.kl.kl_divergence(logits, prev_logits).sum(-1)
            # print('Distillation loss', dist_loss)

            lambda1 = 0 if method_name.endswith('woLL') else 1
            lambda2 = 0 if method_name.endswith('woDL') else 10

            total_loss = 1 * task_loss + lambda1 * link_loss + lambda2 * dist_loss
            # print('Total loss', total_loss)

            total_loss.backward()
            optimizer.step()

            return task_loss, link_loss, dist_loss, total_loss

        @torch.no_grad()
        def test():
            model.eval()
            (_final, logits) = model()

            accs = []
            for _, mask in data('train_mask', 'val_mask', 'test_mask'):
                pred = logits[mask].max(1)[1]
                acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                acc = percentage(acc)
                accs.append(acc)
            return accs

        @torch.no_grad()
        def distillation(best_model):
            best_model.eval()
            final, logits = best_model()

            print('original data.num_edges', data.num_edges)

            new_edge = best_model.conv1.cache["new_edge"]
            new_edge = new_edge[:, :]
            print('new_edge', new_edge, new_edge.shape)
            new_edge_temp1 = new_edge[0].unsqueeze(0)
            new_edge_temp2 = new_edge[1].unsqueeze(0)
            new_edge_homo = torch.cat((new_edge_temp2, new_edge_temp1), dim=0)
            print('new_edge_homo', new_edge_homo, new_edge_homo.shape)
            data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)
            data.edge_index = torch.cat(
                [data.edge_index, new_edge_homo], dim=1)
            print('data.num_edges', data.num_edges)

            # reweight
            dense_adj = to_dense_adj(data.edge_index)[0]
            print('dense_adj', dense_adj, dense_adj.shape)
            dense_adj[dense_adj > 1] = 1
            print('dense_adj (norm)', dense_adj, dense_adj.shape)

            # # drop
            # del_edge = best_model.conv1.cache["del_edge"]
            # print('del_edge', del_edge, del_edge.shape)
            # for i, j in del_edge.T:
            #     dense_adj[i][j] = 0
            #     dense_adj[j][i] = 0

            # sparse
            edge_index, edge_weight = dense_to_sparse(dense_adj)
            print('edge_index', edge_index, edge_index.shape)
            print('edge_weight', edge_weight, edge_weight.shape)
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            print('data.num_edges', data.num_edges)

            pred = logits.max(1)[1]
            Y = F.one_hot(pred).float()
            YYT = torch.matmul(Y, Y.T)
            return final, logits, YYT

        # input('Start traning')
        best_val_acc = test_acc = 0
        best_checkpoiont = {}
        log_training(f'Start Training {z}', trainlog_path)
        log_training(f'*********************************', trainlog_path)
        for epoch in range(1, total_epoch + 1):
            task_loss, link_loss, dist_loss, total_loss = train()
            train_acc, val_acc, tmp_test_acc = test()

            result = {'task_loss': task_loss, 'link_loss': link_loss,
                      'dist_loss': dist_loss, 'total_loss': total_loss,
                      'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': tmp_test_acc}
            wandb.log(result)

            log_text = f'\tEpoch: {epoch} Train: {train_acc} Val: {val_acc} Test: {test_acc}'
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                best_model = GCN().to(device)
                best_model.load_state_dict(model.state_dict())
                best_model.conv1.cache["new_edge"] = model.conv1.cache["new_edge"]
                best_edge, best_weight = data.edge_index, data.edge_attr
                log_text += ' (Best Epoch)'
            log_training(log_text, trainlog_path)
        log_training(
            f'Run: {z} Val. Acc.: {best_val_acc} Test Acc.: {test_acc}', trainlog_path)
        log_training(f'Finished Training {z}', trainlog_path)
        log_training(f'*********************************', trainlog_path)

        val_accs.append(best_val_acc)
        test_accs.append(test_acc)

        final, logits, YYT = distillation(model)

        A = to_dense_adj(data.edge_index)[0]
        A[A > 1] = 1
        A.fill_diagonal_(1)

        # input('Compare toplogy')
        perf_stat = compare_topology(A, data, exp_path/('confmat'+str(z) +
                                                        '.txt'), exp_path/('confmat'+str(z)+'.png'))
        tp_gain = perf_stat['tp'] - prev_tp
        fp_gain = perf_stat['fp'] - prev_fp
        perf_stat['z'] = z
        perf_stat['tp_gain'] = tp_gain
        perf_stat['fp_gain'] = fp_gain
        perf_stat['tp_over_fp'] = round(tp_gain / fp_gain, 4)
        wandb.log(perf_stat)

        for key, val in perf_stat.items():
            wandb.run.summary[key] = val

        # input('TSNE')
        plot_tsne(final, data.y, exp_path/('tsne_'+str(z)+'.png'))

        # input('Plot topology')
        plot_sorted_topology_with_gold_topology(
            A, gold_A, data, exp_path/('topofig'+str(z)+'.png'), sorting=True)

        save_model(model, final, best_edge, best_weight, basemodel_path)
        print()

        wnb_run.finish()

    val_accs = np.array(val_accs)
    mean = np.mean(val_accs)
    std = np.std(val_accs)

    test_accs = np.array(test_accs)
    mean = np.mean(test_accs)
    std = np.std(test_accs)

    superprint(f'Final Val. Acc.: {best_val_acc}', ressum_path)
    superprint(f'Final Test. Acc.: {test_acc}', ressum_path)
    superprint(f'Mean Val. Acc.: {mean} +/- {std}',
               ressum_path, append=False)
    superprint(
        f'Mean Test. Acc.: {round(mean, 3)} +/- {round(std, 3)}', ressum_path)
    superprint(f'Vals Accs: {val_accs}', ressum_path)
    superprint(f'Test Accs {test_accs}', ressum_path)
