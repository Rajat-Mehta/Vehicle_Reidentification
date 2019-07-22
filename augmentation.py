from __future__ import absolute_import

from torchvision.transforms import *

#from PIL import Image
import random
import math
import numpy as np
import torch
from imgaug import augmenters as iaa


class ImgAugTransform:
    def __init__(self, rt=None, tl=None, scale=None, colorspace=None, aug_comb=None):
        self.trans = []
        if rt is not None:
            self.trans = self.trans + [iaa.Affine(rotate=(-rt, rt), mode='constant')]
        if tl is not None:
            self.trans = self.trans + [iaa.Affine(translate_percent={"x": (-tl[0], tl[1]), "y": (-tl[0], tl[1])},
                                                  mode='constant')]
        if scale is not None:
            self.trans = self.trans + [iaa.Affine(scale=(scale[0], scale[1]), mode='constant')]

        if colorspace:
            self.trans = self.trans + [
                iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                iaa.WithChannels(0, iaa.Add((50, 100))),
                iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
            ]

        self.aug = iaa.Sequential(self.trans)

        if aug_comb:
            if rt and tl and scale:
                self.aug = iaa.SomeOf(1, [
                    iaa.Affine(rotate=(-rt, rt), mode='constant'),
                    iaa.Affine(translate_percent={"x": (-tl[0], tl[1]), "y": (-tl[0], tl[1])},
                               mode='constant'),
                    iaa.Affine(scale=(scale[0], scale[1]), mode='constant')
                ], random_order=True)
            else:
                print("Please pass values for rotation, translation and scaling augmentations.")
                exit()

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img


class ColorAugmentation(object):
    """Randomly alters the intensities of RGB channels.
    Reference:
        Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural
        Networks. NIPS 2012.
    Args:
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor
