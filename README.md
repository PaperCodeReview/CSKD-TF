# CSKD-TF
This is an unofficial implementation of CS-KD [(Regularizing Class-wise Predictions via Self-knowledge Distillation)](https://arxiv.org/abs/2003.13964).

## Requirements
- python >= 3.6
- tensorflow >= 2.2

## Training
```
```

## Evaluation
```
```

## Results
Our model achieves the following performance on :
### ResNet-18
|    Dataset    | Top-1 error rates (%) | Top-1 error rates (ours) |
| ------------- | --------------------- | ------------------------ |
|   CIFAR-100   |     21.99 (± 0.13)    |             -            |
| TinyImageNet  |     41.62 (± 0.38)    |             -            |
| CUB-200-2011  |     33.28 (± 0.99)    |             -            |
| Stanford Dogs |     30.85 (± 0.28)    |             -            |
|     MIT67     |     40.45 (± 0.45)    |             -            |

## Citation
```
@InProceedings{Yun_2020_CVPR,
author = {Yun, Sukmin and Park, Jongjin and Lee, Kimin and Shin, Jinwoo},
title = {Regularizing Class-Wise Predictions via Self-Knowledge Distillation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```