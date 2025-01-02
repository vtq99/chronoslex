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
TBD

# Benchmarking
TBD

# Citation
If you find the metric and this repo helpful, please consider cite:
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