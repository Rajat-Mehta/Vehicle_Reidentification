<h1 align="center"> Vehicle re-identification </h1>

This repository contains a baseline model for vehicle reidentificaiton. It consists of multiple model architectures like siamese network, dense net architecture. You can configure which type of architecture to use during training by passing an argument for ex: --use_siamese to use siamese model and if nothing is passed it uses the dense net architecture by default.

These two models will be used as baseline models in my Master Thesis project. Any suggestions or feedback is welcome.

This repository is adapted from Person Reidentification repository which can be found here: https://github.com/layumi/Person_reID_baseline_pytorch
## Table of contents
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    
## Features
Now we have supported:
- Training vehicle reidentification model
- Testing it by feeding query images
- Add custom model and loss function

## Prerequisites

- Python 3.6
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+
- [Optional] apex (for float16) 
- [Optional] [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- [Optinal] You may skip it. Install apex from the source
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0/0.4.0/0.5.0/1.0.0 and Torchvision 0.2.0/0.2.1 .
