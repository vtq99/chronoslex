import torch
import numpy as np
from scipy.stats import sem
from torch.autograd import Variable


def prepare_data(x, y, dataset_name: str):
    x = x.to(dtype=torch.int64).cuda()
    if len(y.shape) > 1:
        y = y.squeeze(1).cuda()
    return x, y


def forward_pass(x, y, dataset, network, criterion):
    logits = network(x)
    if len(y.shape) > 1:
        y = y.squeeze(1)
    loss = criterion(logits.view(-1, dataset.num_classes), y.float().view(-1, dataset.num_classes))

    return loss, logits, y


def split_into_groups(g):
    """
    From https://github.com/p-lambda/wilds/blob/f384c21c67ee58ab527d8868f6197e67c24764d4/wilds/common/utils.py#L40.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts

def get_collate_functions(args, train_dataset):
    # if 'uklex' in args.dataset or 'eurlex' in args.dataset:
    train_collate_fn = collate_fn
    eval_collate_fn = collate_fn

    if 'ecthr' in args.dataset:
        train_collate_fn = ecthr_collate_fn
        eval_collate_fn = ecthr_collate_fn
    # else:
    #     train_collate_fn = None
    #     eval_collate_fn = None

    return train_collate_fn, eval_collate_fn
    # return None, None

def collate_fn(batch):
    # codes = [item[0] for item in batch]
    # target = [item[1] for item in batch]
    # if len(batch[0]) == 2:
    #     return [codes, target]
    # else:
    #     print('COLLATE ERROR?')
    #     # groupid = torch.cat([item[2] for item in batch], dim=0).unsqueeze(1)
    #     return [codes, target]
    return tuple(zip(*batch))


def ecthr_collate_fn(batch):
    max_seg_length= 128
    max_segments = 64
    case_template = [[0] * max_seg_length]
    output = []
    for item in batch:
        item = list(item)
        if len(item[0].shape) != 3:
            item[typ] = item[typ][None, :, :]
        mid = []
        for i in range(2):
            cropped = item[typ][:max_segments]
            updated = cropped[:, :, i].tolist() + case_template * (max_segments - len(cropped[:, :, i]))
            mid.append(torch.Tensor(updated))
        mid = torch.stack(mid, dim=2)
        mid = torch.squeeze(mid, dim=0)
        item[0] = mid
        output.append(tuple(item))

    return tuple(zip(*output))


def default_rprecision_score(y_true, y_score):
    """R-Precision
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    Returns
    -------
    precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) == 1:
        raise ValueError("The score cannot be approximated.")
    elif len(unique_y) > 2:
        raise ValueError("Only supported for two relevant levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:n_pos])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by n_pos such that the best achievable score is always 1.0.
    return float(n_relevant) / n_pos


def mean_rprecision(y_true, y_score):
    """Mean r-precision
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevant labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    Returns
    -------
    mean r-precision : float
    """

    p_ks = []
    for y_t, y_s in zip(y_true, y_score):
        if np.sum(y_t == 1):
            p_ks.append(default_rprecision_score(y_t, y_s))

    return np.mean(p_ks), sem(p_ks)
