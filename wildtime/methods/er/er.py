import os

import numpy as np
import torch
from sklearn import metrics
from scipy.special import expit

from ..base_trainer import BaseTrainer
from ..dataloaders import FastDataLoader
from ..utils import prepare_data, forward_pass


class ER(BaseTrainer):
    """
    Averaged Gradient Episodic Memory (A-GEM)

    Code adapted from https://github.com/aimagelab/mammoth.

    Original Paper:

        @article{chaudhry2018efficient,
        title={Efficient lifelong learning with a-gem},
        author={Chaudhry, Arslan and Ranzato, Marc'Aurelio and Rohrbach, Marcus and Elhoseiny, Mohamed},
        journal={arXiv preprint arXiv:1812.00420},
        year={2018}
        }
    """
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)

        self.replay_freq = args.replay_freq
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'ER-replay_freq={self.replay_freq}-{self.base_trainer_str}'


    def train_step(self, dataloader, timestamp):
        self.network.train()
        loss_all = []
        for step, (x, y) in enumerate(dataloader):
            if self.current_patience == 0:
                print('Stopped at step ' + str(step))
                if self.scheduler is not None:
                    self.scheduler.step()
                break
            x = torch.stack(x)
            y = torch.FloatTensor([[1 if label in labels else 0 for label in range(self.num_classes)] for labels in y])
            x, y = prepare_data(x, y, str(self.train_dataset))

            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion)
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # REPLAY
            if step % self.replay_freq == 0 and self.eval_dataset.ENV.index(timestamp) != 0:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(self.eval_dataset.ENV[self.eval_dataset.ENV.index(timestamp) - 1])
                replay_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                       batch_size=self.mini_batch_size,
                                                       num_workers=self.num_workers,
                                                       collate_fn=self.train_collate_fn)
                for _, (xr, yr) in enumerate(replay_dataloader):
                    self.network.train()
                    xr = torch.stack(xr)
                    yr = torch.FloatTensor(
                        [[1 if label in labels else 0 for label in range(self.num_classes)] for labels in yr])
                    xr, yr = prepare_data(xr, yr, str(self.eval_dataset))
                    loss_r, _, _ = forward_pass(xr, yr, self.eval_dataset, self.network, self.criterion)
                    loss_all.append(loss_r.item())
                    self.optimizer.zero_grad()
                    loss_r.backward()
                    self.optimizer.step()
                    break
                self.eval_dataset.update_current_timestamp(timestamp)

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
