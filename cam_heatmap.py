import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from model import *

parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--data_dir',default='../Datasets/VeRi_with_plate/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet', type=str, help='save model path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--query_index', default=0, type=int, help='test_image_index')

opt = parser.parse_args()
qi = opt.query_index

config_path = os.path.join('./model',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16']
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751


def heatmap2d(img, arr):
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="Image")
    ax1 = fig.add_subplot(122, title="Heatmap")
    img = img.permute(1, 2, 0)
    ax0.imshow(img)
    heatmap = ax1.imshow(arr, cmap='jet')
    fig.colorbar(heatmap)
    #plt.show()
    fig.savefig('heatmap')


def generate_heatmap(model, query):
    with torch.no_grad():
        x = model.model.conv1(query.cuda())
        x = model.model.bn1(x)
        x = model.model.relu(x)
        x = model.model.maxpool(x)
        x = model.model.layer1(x)
        x = model.model.layer2(x)
        output = model.model.layer3(x)
        # output = model.model.layer4(output)
        # print(output.shape)
        heatmap = output.squeeze().sum(dim=0).cpu().numpy()
        # print(heatmap.shape)
        # test_array = np.arange(100 * 100).reshape(100, 100)
        # Result is saved tas `heatmap.png`
        # print(heatmap.shape)

        return heatmap

data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data_transforms_visualize = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor()
])

image_datasets = {x: datasets.ImageFolder(os.path.join(opt.data_dir, x), data_transforms) for x in ['gallery', 'query']}

query_path, _ = image_datasets['query'].imgs[qi]

query_img = Image.open(query_path)
query = data_transforms(query_img)
query_vis = data_transforms_visualize(query_img)
query.unsqueeze_(0)


model = ft_net(class_num=575)
model.load_state_dict(torch.load('./model/ft_ResNet/net_99.pth'))

model.classifier.classifier = nn.Sequential()
model = model.eval().cuda()

heatmap = generate_heatmap(model, query)

heatmap2d(query_vis, heatmap)
