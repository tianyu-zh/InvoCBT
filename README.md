# InvoCBT

This repository is the official implementation of the paper "InvoCBT: CNN and Transformer based framework for
scribble-supervised medical image segmentation."

## Datasets

### ACDC
1. The ACDC dataset with mask annotations can be downloaded from [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/).
2. The scribble annotations of ACDC have been released in [ACDC scribbles](https://vios-s.github.io/multiscale-adversarial-attention-gates/data). 
3. The pre-processed ACDC data used for training could be directly downloaded from [ACDC_dataset](https://github.com/HiLab-git/WSL4MIS/tree/main/data/ACDC).

## Requirements

Some important required packages include:
* Python 3.8
* CUDA 11.3
* [Pytorch](https://pytorch.org) 1.10.1.
* torchvision 0.11.2
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

Follow official guidance to install [Pytorch](https://pytorch.org).

## Training

To train the model, run this command:

```train
python3 train.py --root_path <dataset_path> --exp <path_to_save_model> --linear_layer
```