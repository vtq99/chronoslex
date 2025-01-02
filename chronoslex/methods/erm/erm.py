import os

from ..base_trainer import BaseTrainer


class ERM(BaseTrainer):
    """
    Empirical Risk Minimization
    """
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'ERM-{self.base_trainer_str}'