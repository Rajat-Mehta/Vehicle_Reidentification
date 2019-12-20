<h1 align="center"> Vehicle re-identification </h1>

This repository contains scripts that can be used to train a Vehicle Re-Identificaiton model. It contains scripts to train different model architectures like simple ResNet and VGG based re-identification networks, siamese networks, part-based convolutional baseline (PCB), different variants of PCB (with horizontal, vertical and checkerboard partitioning), cluster-based convolutional baseline (CCB), and a Fusion model that combines the above mentioned models. You can configure which type of architecture to use during training by passing an argument for ex: --use_siamese to use siamese model, --pcb for a PCB model etc.

We use a part-based convolutional baseline (PCB) model as our baseline model that was proposed in the [PCB](https://arxiv.org/pdf/1711.09349.pdf) paper and we adapted it's implementation from the Person Re-Identification repository which can be found here: https://github.com/layumi/Person_reID_baseline_pytorch


## Table of contents
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Installation](#installation)
    
## Features
Things that you can do with this repo:
- Training a vehicle reidentification model using a simple classification network like ResNet, VGG, etc on VeRi-776 and VERI-wild dataset.
- Train a Part-based Convolutional Baseline (PCB) model on VeRi-776 and VERI-wild dataset.
- Train a Part-based Convolutional Baseline model with checkerboard partitioning (PCB-checkerboard) on VeRi-776 and VERI-wild dataset.
- Train a Cluster-based Convolutional Baseline (CCB) model on VeRi-776 and VERI-wild dataset.
- Train a Siamese network using triplet loss function on VeRi-776 and VERI-wild dataset.
- Train a Fusion model consisting of a combination of PCB-checkerboard, CCB and a Siamese network using early fusion technique on VeRi-776 and VERI-wild dataset.
- Test and evaluate all of the above models using the query and gallery image set.
- Use the trained models to perform inference on new images.

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
