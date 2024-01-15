import numpy as np
import torch
import torch.nn as nn
import random

from .networks.lwan import LWANBertClassifier
from .networks.lwan_lora import LWANBertLoRAClassifier
from .networks.lwan_adapter import LWANBertAdapterClassifier
from .networks.hierbert import HierarchicalBert
from .networks.hierbert_lora import HierarchicalBertLoRA
from .networks.hierbert_adapter import HierarchicalBertAdapter

from functools import partial
from .methods.erm.erm import ERM
from .methods.ewc.ewc import EWC
from .methods.er.er import ER
from .methods.agem.agem import AGEM
from .methods.lora.lora import LoRA
from .methods.adapter.adapter import Adapter
from .methods.groupdro.groupdro import GroupDRO
from .methods.irm.irm import IRM
from .methods.coral.coral import DeepCORAL

scheduler = None
group_datasets = ['coral', 'groupdro', 'irm']
print = partial(print, flush=True)


def _uklex18_init(args):
    if args.method in group_datasets:
        from .data.uklex18 import UKLex18Group
        dataset = UKLex18Group(args)
    else:
        from .data.uklex18 import UKLex18
        dataset = UKLex18(args)
    scheduler = None
    criterion = nn.BCEWithLogitsLoss(reduction=args.reduction).cuda()
    if args.method == 'lora':
        network = LWANBertLoRAClassifier(num_classes=dataset.num_classes).cuda()
    elif args.method == 'adapter':
        network = LWANBertAdapterClassifier(num_classes=dataset.num_classes).cuda()
    else:
        network = LWANBertClassifier().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def _uklex69_init(args):
    if args.method in group_datasets:
        from .data.uklex69 import UKLex69Group
        dataset = UKLex69Group(args)
    else:
        from .data.uklex69 import UKLex69
        dataset = UKLex69(args)
    scheduler = None
    criterion = nn.BCEWithLogitsLoss(reduction=args.reduction).cuda()
    if args.method == 'lora':
        network = LWANBertLoRAClassifier(num_classes=dataset.num_classes).cuda()
    elif args.method == 'adapter':
        network = LWANBertAdapterClassifier(num_classes=dataset.num_classes).cuda()
    else:
        network = LWANBertClassifier().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def _eurlex21_init(args):
    if args.method in group_datasets:
        from .data.eurlex21 import EURLex21Group
        dataset = EURLex21Group(args)
    else:
        from .data.eurlex21 import EURLex21
        dataset = EURLex21(args)
    scheduler = None
    criterion = nn.BCEWithLogitsLoss(reduction=args.reduction).cuda()
    if args.method == 'lora':
        network = LWANBertLoRAClassifier(num_classes=dataset.num_classes).cuda()
    elif args.method == 'adapter':
        network = LWANBertAdapterClassifier(num_classes=dataset.num_classes).cuda()
    else:
        network = LWANBertClassifier().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def _eurlex127_init(args):
    if args.method in group_datasets:
        from .data.eurlex127 import EURLex127Group
        dataset = EURLex127Group(args)
    else:
        from .data.eurlex127 import EURLex127
        dataset = EURLex21(args)
    scheduler = None
    criterion = nn.BCEWithLogitsLoss(reduction=args.reduction).cuda()
    if args.method == 'lora':
        network = LWANBertLoRAClassifier(num_classes=dataset.num_classes).cuda()
    elif args.method == 'adapter':
        network = LWANBertAdapterClassifier(num_classes=dataset.num_classes).cuda()
    else:
        network = LWANBertClassifier().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def _ecthr_a_init(args):
    if args.method in group_datasets:
        from .data.ecthr_a import ECtHRAGroup
        dataset = ECtHRAGroup(args)
    else:
        from .data.ecthr_a import ECtHRA
        dataset = ECtHRA(args)
    scheduler = None
    criterion = nn.BCEWithLogitsLoss(reduction=args.reduction).cuda()
    if args.method == 'lora':
        network = HierarchicalBertLoRA(num_classes=dataset.num_classes).cuda()
    elif args.method == 'adapter':
        network = HierarchicalBertAdapter(num_classes=dataset.num_classes).cuda()
    else:
        network = HierarchicalBert().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def _ecthr_b_init(args):
    if args.method in group_datasets:
        from .data.ecthr_b import ECtHRBGroup
        dataset = ECtHRBGroup(args)
    else:
        from .data.ecthr_b import ECtHRB
        dataset = ECtHRB(args)
    scheduler = None
    criterion = nn.BCEWithLogitsLoss(reduction=args.reduction).cuda()
    if args.method == 'lora':
        network = HierarchicalBertLoRA(num_classes=dataset.num_classes).cuda()
    elif args.method == 'adapter':
        network = HierarchicalBertAdapter(num_classes=dataset.num_classes).cuda()
    else:
        network = HierarchicalBert().cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def trainer_init(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(args.device)
    if args.method in ['groupdro', 'irm']:
        args.reduction = 'none'
    else:
        args.reduction = 'mean'
    return globals()[f'_{args.dataset}_init'](args)


def init(args):
    dataset, criterion, network, optimizer, scheduler = trainer_init(args)
    method_dict = {'groupdro': 'GroupDRO', 'coral': 'DeepCORAL', 'irm': 'IRM', 'er': 'ER', 'agem':'AGEM',
                   'erm': 'ERM', 'ewc': 'EWC', 'lora': 'LoRA', 'adapter': 'Adapter'}
    print('Running ' + args.method)
    trainer = globals()[method_dict[args.method]](args, dataset, network, criterion, optimizer, scheduler)
    return trainer


def train(args):
    trainer = init(args)
    trainer.run()
