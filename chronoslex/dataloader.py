scheduler = None
is_group_datasets = ['coral', 'groupdro', 'irm']


def _uklex18_init(args):
    if args.method in group_datasets:
        from .data.uklex18 import UKLex18Group
        dataset = UKLex18Group(args)
    else:
        from .data.uklex18 import UKLex18
        dataset = UKLex18(args)
    return dataset


def _uklex69_init(args):
    if args.method in group_datasets:
        from .data.uklex69 import UKLex69Group
        dataset = UKLex69Group(args)
    else:
        from .data.uklex69 import UKLex69
        dataset = UKLex69(args)
    return dataset


def _eurlex21_init(args):
    if args.method in group_datasets:
        from .data.eurlex21 import EURLex21Group
        dataset = EURLex21Group(args)
    else:
        from .data.eurlex21 import EURLex21
        dataset = EURLex21(args)
    return dataset


def _eurlex127_init(args):
    if args.method in group_datasets:
        from .data.eurlex127 import EURLex127Group
        dataset = EURLex127Group(args)
    else:
        from .data.eurlex127 import EURLex127
        dataset = EURLex21(args)
    return dataset


def _ecthr_a_init(args):
    if args.method in group_datasets:
        from .data.ecthr_a import ECtHRAGroup
        dataset = ECtHRAGroup(args)
    else:
        from .data.ecthr_a import ECtHRA
        dataset = ECtHRA(args)
    return dataset


def _ecthr_b_init(args):
    if args.method in group_datasets:
        from .data.ecthr_b import ECtHRBGroup
        dataset = ECtHRBGroup(args)
    else:
        from .data.ecthr_b import ECtHRB
        dataset = ECtHRB(args)
    return dataset


def getdata(args):
    dataset_name = args.dataset
    if dataset_name == 'uklex18': return _uklex18_init(args)
    if dataset_name == 'uklex69': return _uklex69_init(args)
    if dataset_name == 'eurlex21': return _eurlex21_init(args)
    if dataset_name == 'eurlex127': return _eurlex127_init(args)
    if dataset_name == 'ecthr_a': return _ecthr_a_init(args)
    if dataset_name == 'ecthr_b': return _ecthr_b_init(args)
