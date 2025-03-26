# Knowledge-Driven Prototype Refinement for Few-Shot Fine-Grained Recognition

> **Authors:** 
> Jiale Chen, et al.

This repository provides code for "Knowledge-Driven Prototype Refinement for Few-Shot Fine-Grained Recognition"


## Requirements

 - `Python 3.8`
 - [`Pytorch`](http://pytorch.org/) >= 1.7.0 
 - `Torchvision` = 0.10
 - `scikit-image` = 0.18.1

 
## Vord Vector Preparation
Check and replace the relevant files that store category tags

```bash
python ./makeWordVec/make.py
```

## Data Preparation
Configuring Datasets
When using different datasets, you need to modify the dataset path in the corresponding config file.

check dataset_dir in ./datasets/XXX-dataset changed to the path to the image folder in your environment
check wordpkl_path in ./datasets/XXX-dataset change to the word vector of the corresponding dataset generated in the previous step



## How to run

```bash
python train.py --dataset [type of dataset] --model [backbone] --num_classes [num-classes] --nExemplars [num-shots]
```

## Acknowledgement

This code is based on the implementations of [**fewshot-CAN**](https://github.com/blue-blue272/fewshot-CAN).


