import os

import numpy as np
import torch
from sklearn import metrics
from scipy.special import expit

from .buffer import Buffer
from ..base_trainer import BaseTrainer
from ..dataloaders import FastDataLoader
from ..utils import prepare_data, forward_pass


class AGEM(BaseTrainer):
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

        self.buffer = Buffer(self.args.buffer_size, self._device())
        self.grad_dims = []
        for param in self.network.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self._device())
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self._device())
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'AGEM-buffer_size={self.args.buffer_size}-{self.base_trainer_str}'

    def _device(self):
        return next(self.network.parameters()).device

    def end_task(self, dataloader):
        sample = next(iter(dataloader))
        cur_x, cur_y = sample
        cur_x = torch.stack(cur_x)
        cur_y = torch.FloatTensor([[1 if label in labels else 0 for label in range(self.num_classes)] for labels in cur_y])
        cur_x, cur_y = prepare_data(cur_x, cur_y, str(self.train_dataset))
        self.buffer.add_data(
            examples=cur_x,
            labels=cur_y
        )

    def train_step(self, dataloader, timestamp):
        self.network.train()
        loss_all = []
        for step, (x, y) in enumerate(dataloader):
            if self.current_patience == 0:
                print('Stopped at step ' + str(step))
                if self.scheduler is not None:
                    self.scheduler.step()
                self.end_task(dataloader)
                break
            x = torch.stack(x)
            y = torch.FloatTensor([[1 if label in labels else 0 for label in range(self.num_classes)] for labels in y])
            x, y = prepare_data(x, y, str(self.train_dataset))

            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion)
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()

            if not self.buffer.is_empty():
                store_grad(self.network.parameters, self.grad_xy, self.grad_dims)

                buf_data = self.buffer.get_data(self.mini_batch_size, transform=None)
                if len(buf_data) > 2:
                    buf_inputs = [buf_data[0], buf_data[1]]
                    buf_labels = buf_data[2]
                else:
                    buf_inputs, buf_labels = buf_data
                buf_inputs, buf_labels = prepare_data(buf_inputs, buf_labels, str(self.train_dataset))
                self.network.zero_grad()
                penalty, buff_outputs, buf_labels = forward_pass(buf_inputs, buf_labels, self.train_dataset, self.network,
                                                                 self.criterion)
                penalty.backward()
                store_grad(self.network.parameters, self.grad_er, self.grad_dims)

                dot_prod = torch.dot(self.grad_xy, self.grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                    overwrite_grad(self.network.parameters, g_tilde, self.grad_dims)
                else:
                    overwrite_grad(self.network.parameters, self.grad_xy, self.grad_dims)

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
                self.end_task(dataloader)
                break

def store_grad(params, grads, grad_dims):
    """
    This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger
