# CSKD-TF
This is an unofficial implementation of CS-KD [(Regularizing Class-wise Predictions via Self-knowledge Distillation)](https://arxiv.org/abs/2003.13964).

## Requirements
- python >= 3.6
- tensorflow >= 2.2

## Training
```
python main.py \
    --backbone resnet18 \
    --dataset cifar100 \
    --loss cls \
    --temperature 4 \
    --loss_weight 1 \
    --checkpoint \
    --history \
    --lr_scheduler \
    --src_path /path/for/source \
    --data_path /path/for/data \
    --result_path /path/for/result \
    --gpus 0
```

## Evaluation
```
```

## Results
Our model achieves the following performance on :
### ResNet-18
|    Dataset    | Top-1 error rates (paper, Cross-entropy) | Top-1 error rates (paper, CSKD) | Top-1 error rates (ours, Cross-entropy) | Top-1 error rates (ours, CSKD) |
| ------------- | ---------------------------------------- | ------------------------------- | --------------------------------------- | ------------------------------ |
|   CIFAR-100   |              24.71 (± 0.24)              |          21.99 (± 0.13)         |                   27.73                 |              29.76             |
| TinyImageNet  |              43.53 (± 0.19)              |          41.62 (± 0.38)         |                   48.62                 |              49.35             |
| CUB-200-2011  |              46.00 (± 1.43)              |          33.28 (± 0.99)         |                     -                   |                -               |
| Stanford Dogs |              36.29 (± 0.32)              |          30.85 (± 0.28)         |                     -                   |                -               |
|     MIT67     |              44.75 (± 0.80)              |          40.45 (± 0.45)         |                     -                   |                -               |

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