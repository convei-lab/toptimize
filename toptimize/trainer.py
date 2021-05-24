import torch
import torch.nn.functional as F
from collections import Counter
from copy import deepcopy
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, degree
from torch_geometric.utils.sparse import dense_to_sparse
from utils import (
    percentage,
    log_training,
    compare_topology,
    cold_start
)
from scipy.sparse import csr_matrix
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import statistics
from torch.distributions import Categorical


class Trainer():
    def __init__(self, model, data, device, trainlog_path=None, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.label = data.y
        self.one_hot_label = F.one_hot(data.y).float()
        self.features = data.x
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        self.max_num_nodes = data.num_nodes
        self.adj = to_dense_adj(
            self.edge_index, max_num_nodes=data.num_nodes)[0]
        self.gold_adj = torch.matmul(self.one_hot_label, self.one_hot_label.T)
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        self.device = device
        self.logfile = trainlog_path

        self.checkpoint = {}
        self.final_model = None

    def fit(self, step, total_epoch, lambda1, lambda2, link_pred=None, teacher=None, use_last_epoch=False, use_loss_epoch=False, wnb_run=None, best_final=None, gold_adj=None):
        best_loss, best_acc = 1e10, 0
        self.final_model = self.duplicate(self.model)

        # self.degree_test()
        # selected_nodes = self.select_by_degree()
        if best_final != None:
            selected_nodes = self.select_by_conf(best_final)
        else:
            selected_nodes = None

        log_training(f'Start Training Step {step}', self.logfile)
        log_training(f"{'*'*40}", self.logfile)
        for epoch in range(1, total_epoch + 1):

            self.model.train()
            self.optimizer.zero_grad()

            final, logit = self.model(self.features, self.edge_index, self.edge_attr)

            # task_loss, link_loss, dist_loss = self.loss(
            #     logit, link_pred=link_pred, teacher=teacher, selected=selected_nodes)

            task_loss, link_loss, dist_loss = self.loss(
                logit, final, link_pred=link_pred, teacher=teacher, selected=selected_nodes)

            total_loss = task_loss + lambda1 * link_loss + lambda2 * dist_loss

            for mask in [self.train_mask]:

                pred = logit[mask].max(1)[1]
                acc = pred.eq(self.label[mask]).sum(
                ).item() / mask.sum().item()
                train_acc = percentage(acc)

            total_loss.backward()
            self.optimizer.step()

            (val_acc, tmp_test_acc), logit, final = self.test()
            log_text = f'Epoch: {epoch} Loss: {round(float(total_loss), 4)}  Train: {train_acc} Val: {val_acc} Test: {tmp_test_acc}'

            if wnb_run:
                result = {'task_loss': task_loss, 'link_loss': link_loss,
                          'dist_loss': dist_loss, 'total_loss': total_loss,
                          'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': tmp_test_acc}
                wnb_run.log(result)
            if use_loss_epoch:
                if total_loss < best_loss:
                    best_loss = total_loss
                    test_acc = tmp_test_acc
                    log_text += ' (best epoch)'
                    # print("best_logit_by_loss", logit)
                    # input()
                    self.cache_checkpoint(
                        self.model, logit, epoch, total_loss, train_acc, val_acc, tmp_test_acc)
            else:
                if val_acc > best_acc:
                    best_acc = val_acc
                    test_acc = tmp_test_acc
                    best_final = final
                    log_text += ' (best epoch)'
                    # entropy = Categorical(probs = F.softmax(best_final), dim=1).entropy()
                    # print(entropy, entropy.shape)
                    # input()
                    # print("best_logit_by_acc", logit)
                    # input()
                    self.cache_checkpoint(
                        self.model, logit, epoch, total_loss, train_acc, val_acc, tmp_test_acc, best_final)
            log_training(log_text, self.logfile)

            # if step > 0:
            #     intra_class_score = []
            #     inter_class_score = []
            #     score = self.model.conv1.cache["edge_score"].detach().cpu().numpy()
            #     total_edge_index = self.model.conv1.cache["total_edge_index"]
            #     # print(gold_adj, gold_adj.shape, gold_adj[total_edge_index[0][0]][total_edge_index[1][0]])
            #     for i in range(int(total_edge_index.shape[1]/2), total_edge_index.shape[1]):
            #         row = total_edge_index[0][i]
            #         col = total_edge_index[1][i]
            #         if gold_adj[row][col] == 1:
            #             intra_class_score.append(score[i])
            #         else:
            #             inter_class_score.append(score[i])

            #     # print("score", min(score), max(score))
            #     # print("intra", statistics.mean(intra_class_score))
            #     # print("inter", statistics.mean(inter_class_score))

            #     # intra_class_score.remove(max(intra_class_score))
            #     data = [intra_class_score, inter_class_score]
            #     fig1, ax1 = plt.subplots()
            #     ax1.set_title('Intra-Inter Class Edge Scores Box')
            #     ax1.boxplot(data, vert=False)
            #     plt.savefig("example_0.1.pdf", bbox_inches='tight')
            #     # print("total_edge_index", total_edge_index, total_edge_index.shape, total_edge_index[0][0], total_edge_index[1][0])
            #     input()

        if use_last_epoch:
            self.cache_checkpoint(
                self.model, logit, epoch, total_loss, train_acc, val_acc, tmp_test_acc)
        log_training(f"{'*'*40}", self.logfile)
        log_training(f'Finished Training Step {step}', self.logfile)
        log_training(
            f'Final Epoch {self.final_epoch} Loss {round(float(self.final_total_loss), 4)} Train: {self.final_train_acc} Val: {self.final_val_acc} Test: {self.final_test_acc}', self.logfile)
        log_training(f'', self.logfile)
        return self.final_train_acc, self.final_val_acc, self.final_test_acc

    # def select_by_degree(self):
    #     selected_nodes = []
    #     min_degree = 0

    #     # torch.set_printoptions(edgeitems=50)

    #     node_degree = degree(self.edge_index[0], num_nodes=self.max_num_nodes)

    #     # a = list(self.edge_index[0].detach().cpu().numpy())
    #     # node_degree2 = []
    #     # for i in range(self.max_num_nodes):
    #     #     node_degree2.append(a.count(i))
    #     # # print(node_degree2, len(node_degree2))
    #     # # input()

    #     # for i in range(self.max_num_nodes):
    #     #     if node_degree2[i] > min_degree:
    #     #         selected_nodes.append(i)

    #     for i in range(self.max_num_nodes):
    #         if node_degree[i] > min_degree:
    #             # print(node_degree[i])
    #             selected_nodes.append(i)

    #     # self.train_mask[i] == True or
    #     # print(selected_nodes)

    #     return selected_nodes

    def select_by_conf(self, final):
        selected_nodes = []
        softmax_final = F.softmax(final, dim=1)

        for i in range(7400):
            if softmax_final[i].max() > 0:
                selected_nodes.append(i)

        return selected_nodes

    # def degree_test(self):
    #     ### number of nodes by degree
    #     # node_degree = degree(self.edge_index[0]).detach().cpu().numpy()
    #     # a = Counter(node_degree)
    #     # print(a)

    #     ### make test mask by degree
    #     # node_degree = degree(self.edge_index[0])
    #     # for i in range(self.max_num_nodes):
    #     #     if self.test_mask[i] == True and node_degree[i] > 10:
    #     #         self.test_mask[i] = False

    #     ### Connected Component Analysis
    #     adj = to_dense_adj(self.edge_index, max_num_nodes=self.max_num_nodes)[0].detach().cpu().numpy()
    #     adj = np.asarray(adj)
    #     adj = csr_matrix(adj)
    #     G = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    #     # cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    #     all_wcc = nx.connected_components(G)
    #     wcc_sizes = Counter([len(wcc) for wcc in all_wcc])
    #     size_seq = sorted(wcc_sizes.keys())
    #     size_hist = [wcc_sizes[x] for x in size_seq]

    #     edge_index1 = cold_start(self.edge_index, ratio=0.75)
    #     adj = to_dense_adj(edge_index1, max_num_nodes=self.max_num_nodes)[0].detach().cpu().numpy()
    #     adj = np.asarray(adj)
    #     adj = csr_matrix(adj)
    #     G1 = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    #     # cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    #     all_wcc1 = nx.connected_components(G1)
    #     wcc_sizes1 = Counter([len(wcc) for wcc in all_wcc1])
    #     size_seq1 = sorted(wcc_sizes1.keys())
    #     size_hist1 = [wcc_sizes1[x] for x in size_seq1]

    #     edge_index2 = cold_start(self.edge_index, ratio=0.5)
    #     adj = to_dense_adj(edge_index2, max_num_nodes=self.max_num_nodes)[0].detach().cpu().numpy()
    #     adj = np.asarray(adj)
    #     adj = csr_matrix(adj)
    #     G2 = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    #     # cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    #     all_wcc2 = nx.connected_components(G2)
    #     wcc_sizes2 = Counter([len(wcc) for wcc in all_wcc2])
    #     size_seq2 = sorted(wcc_sizes2.keys())
    #     size_hist2 = [wcc_sizes1[x] for x in size_seq2]

    #     edge_index3 = cold_start(self.edge_index, ratio=0.25)
    #     adj = to_dense_adj(edge_index3, max_num_nodes=self.max_num_nodes)[0].detach().cpu().numpy()
    #     adj = np.asarray(adj)
    #     adj = csr_matrix(adj)
    #     G3 = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph)
    #     # cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    #     all_wcc3 = nx.connected_components(G3)
    #     wcc_sizes3 = Counter([len(wcc) for wcc in all_wcc3])
    #     size_seq3 = sorted(wcc_sizes3.keys())
    #     size_hist3 = [wcc_sizes3[x] for x in size_seq3]

    #     plt.figure(figsize=(16, 12))
    #     plt.clf()
    #     plt.loglog(size_seq, size_hist, 'ro-', markersize=8, label='0 csr')
    #     plt.text(max(size_seq),min(size_hist)+0.1, '91.76%', fontsize=10, c='r')
    #     # plt.annotate('sample', xy=(max(size_seq),max(size_hist)), xytext=(max(size_seq)+1,max(size_hist)+1), arrowprops=dict(facecolor='black', shrink=0.01))
    #     plt.loglog(size_seq3, size_hist3, 'ro--', c='g', marker='s', markersize=8,  label='0.75 csr')
    #     plt.text(max(size_seq3),min(size_hist3)+0.1, '57.16%', fontsize=10, c='g')
    #     # plt.loglog(size_seq2, size_hist2, 'ro-', c='b', marker='s', markersize=10, label='0.50 csr')
    #     # plt.loglog(size_seq3, size_hist3, 'ro--', c='olive', marker='^', markersize=10,  label='0.75 csr')
    #     # plt.plot(size_seq, size_hist, 'ro-', label='Cora')
    #     # plt.title("WCC Size Distribution")
    #     plt.xlabel("Component size")
    #     plt.ylabel("Number of components")
    #     plt.legend(loc=1)
    #     plt.savefig('component.pdf')

    #     # print(cc, sum(cc))
    #     input()
    #     pass

    def duplicate(self, model):
        try:
            copied_model = deepcopy(model)
        except Exception as e:
            copied_model = model.__class__(
                model.nfeat, model.hidden_sizes, model.nclass).to(self.device)
            copied_model.load_state_dict(model.state_dict())
            copied_model.conv1.cache["new_edge"] = model.conv1.cache["new_edge"]
            copied_model.conv1.cache["del_edge"] = model.conv1.cache["del_edge"]

        return copied_model

    def cache_checkpoint(self, model, logit, epoch, total_loss, train_acc, val_acc, test_acc, best_final):
        self.final_model = self.duplicate(model)

        self.checkpoint['model'] = deepcopy(self.final_model.state_dict())
        self.checkpoint['logit'] = logit.clone().detach()
        self.checkpoint['final'] = best_final.clone().detach()
        self.checkpoint['val'] = val_acc
        self.final_epoch = epoch
        self.final_total_loss = total_loss
        self.final_train_acc = train_acc
        self.final_val_acc = val_acc
        self.final_test_acc = test_acc

    def save_model(self, filename, topo_holder):
        self.checkpoint['edge_index'] = topo_holder.edge_index.clone().detach()
        self.checkpoint['edge_attr'] = topo_holder.edge_attr if topo_holder.edge_attr is not None else None
        print('Saving Model '+str('='*40))
        torch.save(self.checkpoint, filename)
        print('Saved as', filename)

    def loss(self, logit, final, link_pred=None, teacher=None, selected=None):
        task_loss = F.nll_loss(
            logit[self.train_mask], self.label[self.train_mask])
        link_loss = link_pred.loss(self.model) if link_pred else 0
        # link_loss = 0
        dist_loss = 0 if teacher is None else F.kl_div(
            F.log_softmax(final[selected], dim=1), F.softmax(teacher[selected], dim=1), reduction='none').mean()
        # if teacher is not None:
        #     print("logit+teacher_original", logit, logit.shape, teacher, teacher.shape)
        #     print("logit+teacher[selected]", logit[selected], logit[selected].shape, teacher[selected], teacher[selected].shape)
        #     input()
        # dist_loss = 0 if teacher is None else F.kl_div(
        #     logit, teacher, reduction='none', log_target=True).mean()
        # dist_loss = torch.distributions.kl.kl_divergence(logit, prev_logit).sum(-1)

        return task_loss, link_loss, dist_loss

    @ torch.no_grad()
    def test(self):
        self.model.eval()

        with torch.no_grad():
            final, logit = self.model(
                self.features, self.edge_index, self.edge_attr)

            accs = []
            for mask in [self.val_mask, self.test_mask]:
                pred = logit[mask].max(1)[1]
                acc = pred.eq(self.label[mask]).sum(
                ).item() / mask.sum().item()
                acc = percentage(acc)
                accs.append(acc)

        return accs, logit, final

    @ torch.no_grad()
    def ensemble(self, run_dir):
        self.model.eval()
        with torch.no_grad():
            logit_list = []
            for file in run_dir.iterdir():
                if file.suffix == '.pt' and not file.stem.endswith('0'):
                    logit_list.append(torch.load(file)['logit'])

            logits_sum = 0
            for i in range(0, len(logit_list)):
                logits_sum += logit_list[i]
            logit = logits_sum / len(logit_list)

            accs = []
            for mask in [self.train_mask, self.val_mask, self.test_mask]:
                pred = logit[mask].max(1)[1]
                acc = pred.eq(self.label[mask]).sum(
                ).item() / mask.sum().item()
                acc = percentage(acc)
                accs.append(acc)
        return accs

    @ torch.no_grad()
    def infer(self):
        self.final_model.eval()
        with torch.no_grad():
            final, logit = self.final_model(
                self.features, self.edge_index, self.edge_attr)
        return final, logit

    @ torch.no_grad()
    def augment_topology(self, drop_edge=False, eval_new_edge=False):
        edge_index = self.edge_index.clone()
        before_edge_num = edge_index.size(1)

        if self.model.conv1.cache["new_edge"] != None:

            new_edge = self.model.conv1.cache["new_edge"]
            # print('aug_new_edge', new_edge, new_edge.shape)
            # input()
            reversed_edge = torch.zeros_like(new_edge)
            reversed_edge[[1, 0]] = new_edge
            new_edge = torch.cat([new_edge, reversed_edge], dim=1)

            # Addition & Reweight
            edge_index = torch.cat([edge_index, new_edge], dim=1)
            adj = to_dense_adj(edge_index, max_num_nodes=self.max_num_nodes)[0]
            adj[adj > 1] = 1

            if new_edge.size(1) == 0:
                new_adj = torch.zeros_like(adj)
            else:
                new_adj = to_dense_adj(
                    new_edge, max_num_nodes=self.max_num_nodes)[0]

            # Drop
            if drop_edge == True and self.model.conv1.cache["del_edge"] != None:
                if self.model.conv1.cache["new_edge"].shape[1] > self.model.conv1.cache["del_edge"].shape[1]:
                    del_edge = self.model.conv1.cache["del_edge"]
                    adj[del_edge[0], del_edge[1]] = 0
                    adj[del_edge[1], del_edge[0]] = 0
                else:
                    del_edge = None
            if self.model.conv1.cache["del_edge"] == None:
                del_edge = None

            edge_index, edge_attr = dense_to_sparse(adj)
            self.edge_index = edge_index
            self.edge_attr = edge_attr

            log_training(f'New Edge: {new_edge}', self.logfile)
            log_training(f'# New Edge: {new_edge.size(1)}', self.logfile)
            if drop_edge:
                log_training(f'Deleted Edge: {del_edge}', self.logfile)
                if self.model.conv1.cache["del_edge"] != None:
                    if self.model.conv1.cache["new_edge"].shape[1] > self.model.conv1.cache["del_edge"].shape[1]:
                        log_training(
                            f'# Deleted Edge: {del_edge.size(1)}', self.logfile)
                        log_training(
                            f'# Plus-minus: {new_edge.size(1)-del_edge.size(1)}', self.logfile)
            log_training(f'# Before Edge: {before_edge_num}', self.logfile)
            log_training(f'# After Edge: {edge_index.size(1)}', self.logfile)

        else:
            new_edge = None

            # Addition & Reweight
            adj = to_dense_adj(edge_index, max_num_nodes=self.max_num_nodes)[0]
            adj[adj > 1] = 1
            new_adj = adj

            edge_index, edge_attr = dense_to_sparse(adj)
            self.edge_index = edge_index
            self.edge_attr = edge_attr

            log_training(f'New Edge: {new_edge}', self.logfile)
            log_training(f'# New Edge: {0}', self.logfile)
            log_training(f'# Before Edge: {before_edge_num}', self.logfile)
            log_training(f'# After Edge: {edge_index.size(1)}', self.logfile)

        return edge_index, edge_attr, adj, new_edge, new_adj

    # @ torch.no_grad()
    # def augment_topology(self, drop_edge=False, eval_new_edge=False):
    #     edge_index = self.edge_index.clone()
    #     before_edge_num = edge_index.size(1)

    #     if self.final_model.conv1.cache["new_edge"] != None:

    #         new_edge = self.final_model.conv1.cache["new_edge"]
    #         print('aug_new_edge', new_edge, new_edge.shape)
    #         reversed_edge = torch.zeros_like(new_edge)
    #         reversed_edge[[1, 0]] = new_edge
    #         new_edge = torch.cat([new_edge, reversed_edge], dim=1)

    #         # Addition & Reweight
    #         edge_index = torch.cat([edge_index, new_edge], dim=1)
    #         adj = to_dense_adj(edge_index, max_num_nodes=self.max_num_nodes)[0]
    #         adj[adj > 1] = 1

    #         if new_edge.size(1) == 0:
    #             new_adj = torch.zeros_like(adj)
    #         else:
    #             new_adj = to_dense_adj(
    #                 new_edge, max_num_nodes=self.max_num_nodes)[0]

    #         # Drop
    #         if drop_edge == True and self.final_model.conv1.cache["del_edge"] != None:
    #             if self.final_model.conv1.cache["new_edge"].shape[1] > self.final_model.conv1.cache["del_edge"].shape[1]:
    #                 del_edge = self.final_model.conv1.cache["del_edge"]
    #                 adj[del_edge[0], del_edge[1]] = 0
    #                 adj[del_edge[1], del_edge[0]] = 0
    #             else:
    #                 del_edge = None
    #         if self.final_model.conv1.cache["del_edge"] == None:
    #             del_edge = None

    #         edge_index, edge_attr = dense_to_sparse(adj)
    #         self.edge_index = edge_index
    #         self.edge_attr = edge_attr

    #         log_training(f'New Edge: {new_edge}', self.logfile)
    #         log_training(f'# New Edge: {new_edge.size(1)}', self.logfile)
    #         if drop_edge:
    #             log_training(f'Deleted Edge: {del_edge}', self.logfile)
    #             if self.final_model.conv1.cache["del_edge"] != None:
    #                 if self.final_model.conv1.cache["new_edge"].shape[1] > self.final_model.conv1.cache["del_edge"].shape[1]:
    #                     log_training(
    #                         f'# Deleted Edge: {del_edge.size(1)}', self.logfile)
    #                     log_training(
    #                         f'# Plus-minus: {new_edge.size(1)-del_edge.size(1)}', self.logfile)
    #         log_training(f'# Before Edge: {before_edge_num}', self.logfile)
    #         log_training(f'# After Edge: {edge_index.size(1)}', self.logfile)

    #     else:
    #         new_edge = None

    #         # Addition & Reweight
    #         adj = to_dense_adj(edge_index, max_num_nodes=self.max_num_nodes)[0]
    #         adj[adj > 1] = 1
    #         new_adj = adj

    #         edge_index, edge_attr = dense_to_sparse(adj)
    #         self.edge_index = edge_index
    #         self.edge_attr = edge_attr

    #         log_training(f'New Edge: {new_edge}', self.logfile)
    #         log_training(f'# New Edge: {0}', self.logfile)
    #         log_training(f'# Before Edge: {before_edge_num}', self.logfile)
    #         log_training(f'# After Edge: {edge_index.size(1)}', self.logfile)

    #     return edge_index, edge_attr, adj, new_edge, new_adj
