import os

import numpy as np
import torch
from sklearn import metrics
from scipy.special import expit

from ..base_trainer import BaseTrainer
from ..dataloaders import FastDataLoader
from ..utils import prepare_data, forward_pass, split_into_groups


class DeepCORAL(BaseTrainer):
    """
    Deep CORAL

    This algorithm was originally proposed as an unsupervised domain adaptation algorithm.

    Original paper:
        @inproceedings{sun2016deep,
          title={Deep CORAL: Correlation alignment for deep domain adaptation},
          author={Sun, Baochen and Saenko, Kate},
          booktitle={European Conference on Computer Vision},
          pages={443--450},
          year={2016},
          organization={Springer}
        }

    Code adapted from https://github.com/p-lambda/wilds/blob/main/examples/algorithms/deepCORAL.py.
    """

    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.coral_lambda = args.coral_lambda
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'DeepCORAL-coral_lambda={self.coral_lambda}-{self.base_trainer_str}'

    def coral_penalty(self, x, y):
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def train_step(self, dataloader, timestamp):
        self.network.train()
        loss_all = []
        for step, (x, y, g) in enumerate(dataloader):
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
            unique_groups, group_indices, _ = split_into_groups(g)
            n_groups_per_batch = unique_groups.numel()

            classification_loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion)
            coral_loss = torch.zeros(1).cuda()
            for i_group in range(n_groups_per_batch):
                for j_group in range(i_group + 1, n_groups_per_batch):
                    coral_loss += self.coral_penalty(logits[group_indices[i_group]].squeeze(0),
                                                     logits[group_indices[j_group]].squeeze(0))
            if n_groups_per_batch > 1:
                coral_loss /= (n_groups_per_batch * (n_groups_per_batch - 1) / 2)

            loss = classification_loss + self.coral_lambda * coral_loss
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
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
                    with torch.no_grad():
                        logits_val = self.network(x_val)
                        val_loss = self.criterion(logits_val.view(-1, self.train_dataset.num_classes),
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
