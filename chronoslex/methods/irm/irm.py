import os

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from scipy.special import expit

from ..base_trainer import BaseTrainer
from ..dataloaders import InfiniteDataLoader, FastDataLoader
from ..groupdro.loss import LossComputer
from ..utils import prepare_data, forward_pass, split_into_groups


class IRM(BaseTrainer):
    """
    Invariant risk minimization.

    Original paper:
        @article{arjovsky2019invariant,
          title={Invariant risk minimization},
          author={Arjovsky, Martin and Bottou, L{\'e}on and Gulrajani, Ishaan and Lopez-Paz, David},
          journal={arXiv preprint arXiv:1907.02893},
          year={2019}
        }

    Code adapted from https://github.com/p-lambda/wilds/blob/main/examples/algorithms/IRM.py.
    """
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.update_count = 0
        self.irm_lambda = args.irm_lambda
        self.irm_penalty_anneal_iters = args.irm_penalty_anneal_iters
        self.scale = torch.tensor(1.).requires_grad_()
        dataset.current_time = dataset.ENV[0]
        self.loss_computer = LossComputer(self.train_dataset, criterion, is_robust=True)
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'IRM-irm_lambda={self.irm_lambda}-irm_penalty_anneal_iters={self.irm_penalty_anneal_iters}' \
               f'-{self.base_trainer_str}'

    def irm_penalty(self, losses):
        grad_1 = torch.autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = torch.autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def train_step(self, dataloader, timestamp):
        self.network.train()
        loss_all = []
        for step, (x, y, g) in enumerate(dataloader):
            self.network.train()
            if self.current_patience == 0:
                print('Stopped at step ' + str(step))
                if self.scheduler is not None:
                    self.scheduler.step()
                break
            x = torch.stack(x)
            y = torch.FloatTensor([[1 if label in labels else 0 for label in range(self.num_classes)] for labels in y])
            x, y = prepare_data(x, y, str(self.train_dataset))
            g = torch.stack(g)
            g = g.squeeze(1).cuda()

            self.network.zero_grad()
            unique_groups, group_indices, _ = split_into_groups(g)
            n_groups_per_batch = unique_groups.numel()
            avg_loss = 0.
            penalty = 0.
            _, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion)
            for i_group in group_indices:
                group_losses = self.criterion(self.scale * logits[i_group], y[i_group])
                if group_losses.numel() > 0:
                    avg_loss += group_losses.mean()
                penalty += self.irm_penalty(group_losses)
            avg_loss /= n_groups_per_batch
            penalty /= n_groups_per_batch

            if self.update_count >= self.irm_penalty_anneal_iters:
                penalty_weight = self.irm_lambda
            else:
                penalty_weight = 1.0

            loss = avg_loss + penalty * penalty_weight
            loss_all.append(loss.item())
            loss.backward()
            self.optimizer.step()

            # EVAL
            if step % self.eval_freq == 0 and step > 0:
                self.network.eval()
                val_loss_all = []
                pred_all = []
                y_all = []
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)

                test_time_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                      batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                for _, sample in enumerate(test_time_dataloader):
                    if len(sample) == 3:
                        x_val, y_val, _ = sample
                    else:
                        x_val, y_val = sample
                    x_val = torch.stack(x_val)
                    y_val = torch.FloatTensor(
                        [[1 if label in labels else 0 for label in range(self.num_classes)] for labels in y_val])
                    x_val, y_val = prepare_data(x_val, y_val, str(self.eval_dataset))
                    val_criterion = nn.BCEWithLogitsLoss(reduction='mean').cuda()
                    with torch.no_grad():
                        logits_val = self.network(x_val)
                        val_loss = val_criterion(logits_val.view(-1, self.train_dataset.num_classes),
                                                  y_val.float().view(-1, self.train_dataset.num_classes))
                        val_loss_all.append(val_loss.item())

                        pred = (expit(logits_val.detach().cpu().numpy()) > 0.5).astype(int)
                        pred_all = list(pred_all) + pred.tolist()
                        y_all = list(y_all) + y_val.cpu().numpy().tolist()

                val_loss = np.mean(val_loss_all)
                pred_all = np.array(pred_all)
                y_all = np.array(y_all)
                if 'ecthr' in str(self.eval_dataset):
                    y_true = np.zeros((y_all.shape[0], y_all.shape[1] + 1), dtype=np.int32)
                    y_true[:, :-1] = y_all
                    y_true[:, -1] = (np.sum(y_all, axis=1) == 0).astype('int32')
                    y_all = y_true
                    y_pred = np.zeros((pred_all.shape[0], pred_all.shape[1] + 1), dtype=np.int32)
                    y_pred[:, :-1] = pred_all
                    y_pred[:, -1] = (np.sum(pred_all, axis=1) == 0).astype('int32')
                    pred_all = y_pred
                # accuracy = metrics.accuracy_score(y_all, pred_all)
                f1_score_macro = metrics.f1_score(y_all, pred_all, average='macro')
                val_loss = 1 - f1_score_macro

                self.eval_dataset.mode = 2
                self.network.train()
                # EARLY STOPPING
                if self.best_val is None:
                    is_best = True
                else:
                    is_best = val_loss < self.best_val
                if is_best or step < 3 * self.eval_freq:
                    # Reset patience
                    self.current_patience = self.patience
                    self.best_val = val_loss
                    self.save_model(timestamp)
                else:
                    # Decrease patience
                    self.current_patience -= 1

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                break

    def run_online(self):
        for i, t in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.load_model:
                self.load_model(t)
            else:
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.train = True
                time_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(time_dataloader)
                self.save_model(t)
                self.train_dataset.update_historical(i + 1)

            self.eval_dataset.train = False
            avg_acc, worst_acc, best_acc = self.evaluate_stream(i + 1)
            self.task_accuracies[t] = avg_acc
            self.worst_time_accuracies[t] = worst_acc
            self.best_time_accuracies[t] = best_acc
