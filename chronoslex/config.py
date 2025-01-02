
config = {
    'dataset': 'uklex18', # choices=['uklex18', 'uklex69', 'eurlex21', 'eurlex127', 'ecthr_a', 'ecthr_b']
    'method': 'erm', # choices=['erm', 'ewc', 'er', 'agem', 'lora', 'adapter', 'coral', 'irm', 'groupdro']
    'device': 0,  # 'gpu id'
    'random_seed': 1,  # 'random seed number'

    # Training hyperparameters
    'train_update_iter': 10,  # 'train update iter'
    'lr': 0.01,  # 'the base learning rate of the generator'
    'momentum': 0.9,  # 'momentum'
    'weight_decay': 0.0,  # 'weight decay'
    'mini_batch_size': 32,  # 'mini batch size for SGD'
    'reduced_train_prop': None,  # 'proportion of samples allocated to train at each time step'
    'reduction': 'mean',

    # Evaluation
    'offline': False,  # help='evaluate offline at a single time step split'
    'difficulty': False,  # 'task difficulty'
    # todo: set value of split_time
    'split_time': 0,  # 'timestep to split ID vs OOD' #
    'eval_next_timesteps': 1,  # 'number of future timesteps to evaluate on'
    'eval_worst_time': False,  # 'evaluate worst timestep accuracy'
    'load_model': False,  # 'load trained model for evaluation only'
    'eval_metric': 'acc',  # choices=['acc', 'f1', 'rmse']
    'eval_all_timesteps': False,  # 'evaluate at ID and OOD time steps'

    # GroupDRO
    'num_groups': 4,  # 'number of windows for Invariant Learning baselines'
    'group_size': 4,  # 'window size for Invariant Learning baselines'
    'non_overlapping': False,  # 'non-overlapping time windows'

    # EWC
    'ewc_lambda': 1.0,  # help='how strong to weigh EWC-loss ("regularisation strength")'
    'gamma': 1.0,  # help='decay-term for old tasks (contribution to quadratic term)'
    'online': False,  # help='"online" (=single quadratic term) or "offline" (=quadratic term per task) EWC'
    'fisher_n': None,  # help='sample size for estimating FI-matrix (if "None", full pass over dataset)'
    'emp_FI': False,  # help='if True, use provided labels to calculate FI ("empirical FI"); else predicted labels'

    # A-GEM
    'buffer_size': 100,  # 'buffer size for A-GEM'

    # CORAL
    'coral_lambda': 1.0,  # 'how strong to weigh CORAL loss'

    # IRM
    'irm_lambda': 1.0,  # 'how strong to weigh IRM penalty loss'
    'irm_penalty_anneal_iters': 0,  # 'number of iterations after which we anneal IRM penalty loss'

    # Logging, saving, and testing options
    'data_dir': './WildTime/datasets',  # 'directory for datasets.'
    'log_dir': './checkpoints',  # 'directory for summaries and checkpoints.'
    'results_dir': './results',  # 'directory for summaries and checkpoints.'
    'num_workers': 0  # 'number of workers in data generator'
}
