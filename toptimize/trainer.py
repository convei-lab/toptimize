import torch
import torch.nn.functional as F

from copy import deepcopy
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.sparse import dense_to_sparse
from utils import (
    percentage,
    log_training,
)


class Trainer():
    def __init__(self, model, data, device, trainlog_path, use_last_epoch, optimizer=None):
        self.model = model
        self.optimizer = optimizer
        self.label = data.y
        self.one_hot_label = F.one_hot(data.y).float()
        self.features = data.x
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        self.adj = to_dense_adj(self.edge_index)[0]
        self.gold_adj = torch.matmul(self.one_hot_label, self.one_hot_label.T)
        self.train_mask = data.train_mask
        self.val_mask = data.val_mask
        self.test_mask = data.test_mask
        self.device = device
        self.logfile = trainlog_path
        self.use_last_epoch = use_last_epoch

        self.checkpoint = {}
        self.final_model = None

    def train(self, step, total_epoch, lambda1, lambda2, link_pred=None, teacher=None, wnb_run=None):

        best_loss = 1e10
        best_logit = 0
        self.final_model = self.model

        log_training(f'Start Training Step {step}', self.logfile)
        log_training(f"{'*'*40}", self.logfile)
        for epoch in range(1, total_epoch + 1):

            self.model.train()
            self.optimizer.zero_grad()
            final, logit = self.model(
                self.features, self.edge_index, self.edge_attr)

            task_loss, link_loss, dist_loss = self.loss(
                logit, link_pred=link_pred, teacher=teacher)

            total_loss = task_loss + lambda1 * link_loss + lambda2 * dist_loss
            total_loss.backward()
            self.optimizer.step()

            train_acc, val_acc, tmp_test_acc = self.test()
            log_text = f'Epoch: {epoch} Loss: {round(float(total_loss), 4)}  Train: {train_acc} Val: {val_acc} Test: {tmp_test_acc}'

            if wnb_run:
                result = {'task_loss': task_loss, 'link_loss': link_loss,
                          'dist_loss': dist_loss, 'total_loss': total_loss,
                          'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': tmp_test_acc}
                wnb_run.log(result)

            if total_loss < best_loss:
                best_loss = total_loss
                test_acc = tmp_test_acc
                best_logit = logit
                log_text += ' (best epoch)'
                self.cache_checkpoint(
                    self.model, logit, epoch, train_acc, val_acc, tmp_test_acc)
            log_training(log_text, self.logfile)
        if self.use_last_epoch:
            self.cache_checkpoint(
                self.model, logit, epoch, train_acc, val_acc, tmp_test_acc)
        log_training(f"{'*'*40}", self.logfile)
        log_training(f'Finished Training Step {step}', self.logfile)
        log_training(
            f'Final Epoch {self.final_epoch} Train: {self.final_train_acc} Val: {self.final_val_acc} Test: {self.final_test_acc}', self.logfile)
        
        log_training(f'', self.logfile)

        return self.final_train_acc, self.final_val_acc, self.final_test_acc, best_logit

    def cache_checkpoint(self, model, logit, epoch, train_acc, val_acc, test_acc):
        self.checkpoint['model'] = model.state_dict()
        self.checkpoint['logit'] = logit.detach().clone()
        self.final_epoch = epoch
        self.final_train_acc = train_acc
        self.final_val_acc = val_acc
        self.final_test_acc = test_acc

        try:
            self.final_model = deepcopy(model)
        except Exception as e:
            # print('Error while caching final model')
            final_model = model.__class__(
                model.nfeat, model.hidden_sizes, model.nclass).to(self.device)
            final_model.load_state_dict(model.state_dict())
            final_model.conv1.cache["new_edge"] = model.conv1.cache["new_edge"]
            final_model.conv1.cache["del_edge"] = model.conv1.cache["del_edge"]
            self.final_model = final_model

    def save_model(self, filename, topo_holder):
        self.checkpoint['edge_index'] = topo_holder.edge_index
        self.checkpoint['edge_attr'] = topo_holder.edge_attr
        self.checkpoint['adj'] = to_dense_adj(topo_holder.edge_index)[0]

        print('Saving Model '+str('='*40))
        torch.save(self.checkpoint, filename)
        print('Saved as', filename)

    def loss(self, logit, link_pred=None, teacher=None):
        task_loss = F.nll_loss(
            logit[self.train_mask], self.label[self.train_mask])
        link_loss = link_pred.loss(self.model) if link_pred else 0
        dist_loss = 0 if teacher is None else F.kl_div(
            logit, teacher, reduction='none', log_target=True).mean()
        # dist_loss = torch.distributions.kl.kl_divergence(logit, prev_logit).sum(-1)

        return task_loss, link_loss, dist_loss

    @ torch.no_grad()
    def test(self, ensemble=False, ensemble_logits=None):
        self.model.eval()
        if ensemble == False:
            final, logit = self.model(
                self.features, self.edge_index, self.edge_attr)

            accs = []
            for mask in [self.train_mask, self.val_mask, self.test_mask]:
                pred = logit[mask].max(1)[1]
                acc = pred.eq(self.label[mask]).sum().item() / mask.sum().item()
                acc = percentage(acc)
                accs.append(acc)
        else:
            logits_sum = 0
            for i in range(0,len(ensemble_logits)):
                logits_sum += ensemble_logits[i]
            logits = logits_sum / len(ensemble_logits)
            accs = []
            for mask in [self.train_mask, self.val_mask, self.test_mask]:
                if mask.sum() == 0:
                    accs.append(0)
                else:
                    pred = logits[mask].max(1)[1]
                    acc = pred.eq(self.label[mask]).sum().item() / mask.sum().item()
                    acc = percentage(acc)
                    accs.append(acc)  

        return accs

    @ torch.no_grad()
    def infer(self):
        self.final_model.eval()
        final, logit = self.final_model(
            self.features, self.edge_index, self.edge_attr)
        return final, logit

    @ torch.no_grad()
    def augment_topology(self, drop_edge=False):
        edge_index = self.edge_index.clone()
        before_edge_num = edge_index.size(1)

        new_edge = self.final_model.conv1.cache["new_edge"]
        reversed_edge = torch.zeros_like(new_edge)
        reversed_edge[[1, 0]] = new_edge
        new_edge = torch.cat([new_edge, reversed_edge], dim=1)

        # Addition & Reweight
        edge_index = torch.cat([edge_index, new_edge], dim=1)
        adj = to_dense_adj(edge_index)[0]
        adj[adj > 1] = 1

        # Drop
        if drop_edge:
            del_edge = self.final_model.conv1.cache["del_edge"]
            adj[del_edge[0], del_edge[1]] = 0
            adj[del_edge[1], del_edge[0]] = 0

        edge_index, edge_attr = dense_to_sparse(adj)
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        log_training(f'New Edge: {new_edge}', self.logfile)
        log_training(f'# New Edge: {new_edge.size(1)}', self.logfile)
        if drop_edge:
            log_training(f'Deleted Edge: {del_edge}', self.logfile)
            log_training(f'# Deleted Edge: {del_edge.size(1)}', self.logfile)
            log_training(
                f'# Plus-minus: {new_edge.size(1)-del_edge.size(1)}', self.logfile)
        log_training(f'# Before Edge: {before_edge_num}', self.logfile)
        log_training(f'# After Edge: {edge_index.size(1)}', self.logfile)

        return edge_index, edge_attr, adj
