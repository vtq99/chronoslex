# ChronosLex
This is the repository for ChronosLex, an incremental training paradigm introduced in \
[ChronosLex: Time-aware Incremental Training for Temporal Generalization of Legal Classification Tasks](https://aclanthology.org/2024.acl-long.166/) \
Santosh T.Y.S.S, Tuan-Quang Vuong, Matthias Grabmair \
ACL 2024

# Introduction
ChronosLex is an incremental training paradigm that trains models on chronological splits, preserving the temporal order of the data.
However, this incremental approach raises concerns about overfitting to recent data, prompting an assessment of mitigation strategies using continual learning and temporal invariant methods.
Our experimental results over six legal multi-label text classification datasets reveal that continual learning methods prove effective in preventing overfitting thereby enhancing temporal generalizability, while temporal invariant methods struggle to capture these dynamics of temporal shifts.

# How to run
We deeply customized the code base of [WILDS](https://wilds.stanford.edu) and [WildTime](https://wild-time.github.io) to fit to our multi-label multi-class text classification tasks.
We also included additional training stategies: ER, LoRA, Bottleneck Adpater on different models: BERT-LWAN, Hierachical BERT.

### Configuration
We use Munch to load the configurations from a dictionary.
```
from munch import DefaultMunch
configs = DefaultMunch.fromDict(config)
```
Parameters in the dictionary ```config``` should include:
- One of the ```'dataset'``` from choices ```['uklex18', 'uklex69', 'eurlex21', 'eurlex127', 'ecthr_a', 'ecthr_b']```
- One of the ```'method'``` from choices ```['erm', 'ewc', 'er', 'agem', 'lora', 'adapter', 'coral', 'irm', 'groupdro']```.
- To use Eval-Fix (our main focus), set ```'eval_fix': True```.
```'split_time'``` specifies the split time step used for training and testing.
- To use Eval-Stream, set ```'eval_next_timesteps'``` to define the number of future timesteps to evaluate on.
- The number of training iterations is controlled by ```'train_update_iters'```.
Training hyperparameters also include
```'momentum'``` of Adam optimizer,
```'weight_decay'```,
```mini_batch_size'``` for SGD,
```'reduction'``` of loss functions,
```'eval_freq'``` (validation frequency),
```'patience'``` for early stopping,
and method-specific hyperparameters. 
- Logging, saving, and testing destinations can be specified with ```'--data_dir'```, ```'--log_dir'``` and ```'--result_dir'```.

### Training
To train a model with the defined configurations, use the code:
```
from chronoslex import baseline_trainer
baseline_trainer.train(configs)
```
To reproduce our results, please refer to Appendix A of our paper.

# Benchmarking
(To-be-updated) In the meantime, please refer to Table 2 and 3, as well as Figure 2 and 3 in our paper.

# Citation
If you find the paper and this repository helpful, please cite:
```
@inproceedings{t-y-s-s-etal-2024-chronoslex,
    title = "{C}hronos{L}ex: Time-aware Incremental Training for Temporal Generalization of Legal Classification Tasks",
    author = "T.y.s.s, Santosh  and
      Vuong, Tuan-Quang  and
      Grabmair, Matthias",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.166/",
    doi = "10.18653/v1/2024.acl-long.166",
    pages = "3022--3039",
    abstract = "This study investigates the challenges posed by the dynamic nature of legal multi-label text classification tasks, where legal concepts evolve over time. Existing models often overlook the temporal dimension in their training process, leading to suboptimal performance of those models over time, as they treat training data as a single homogeneous block. To address this, we introduce ChronosLex, an incremental training paradigm that trains models on chronological splits, preserving the temporal order of the data. However, this incremental approach raises concerns about overfitting to recent data, prompting an assessment of mitigation strategies using continual learning and temporal invariant methods. Our experimental results over six legal multi-label text classification datasets reveal that continual learning methods prove effective in preventing overfitting thereby enhancing temporal generalizability, while temporal invariant methods struggle to capture these dynamics of temporal shifts."
}
```