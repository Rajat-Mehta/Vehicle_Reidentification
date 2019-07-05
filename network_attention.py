from mpl_toolkits.axes_grid1 import make_axes_locatable
from augmentation import RandomErasing
import cv2
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
import scipy.io
import yaml
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from model import *
from cam_heatmap import *

parser = argparse.ArgumentParser(description='Visualizing')

parser.add_argument('--data_dir', default='../Datasets/VeRi_with_plate/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet', type=str, help='save model path')
parser.add_argument('--batchsize', default=512, type=int, help='batchsize')
parser.add_argument('--query_index', default=0, type=int, help='test_image_index')

opt = parser.parse_args()

qi = opt.query_index

config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.fp16 = config['fp16']
opt.PCB = config['PCB']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']

opt.h = 256
opt.w = 128

if 'nclasses' in config:  # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751


def get_model():
    model = ft_net(class_num=575)
    model.load_state_dict(torch.load('./model/ft_ResNet/net_99.pth'))
    model.classifier.classifier = nn.Sequential()
    model = model.eval().cuda()

    return model


def load_features():
    result = scipy.io.loadmat('./model/' + opt.name + '/pytorch_result_VeRi_99.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    return gallery_feature, gallery_cam, gallery_label, query_feature, query_cam, query_label


def get_index(gallery_feature, gallery_cam, gallery_label, query):
    query = query.view(-1, 1)
    score = torch.mm(gallery_feature.cuda(), query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    return index


def imshow(path, title=None, query_vis=None, cax=None):
    """Imshow for Tensor."""
    im = plt.imread(path)

    if title == 'hmap' or title == 'hmap_erased':
        heatmap = plt.imshow(query_vis, cmap='jet')
        # if title == 'hmap':
            # cb = plt.colorbar(heatmap, orientation='horizontal')
            # cb.set_ticks([])
    elif query_vis is not None:
        query_vis = query_vis.permute(1, 2, 0)
        plt.imshow(query_vis)

    else:
        plt.imshow(im)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def evaluate(model, heatmap, query_img, query_vis, j, img_num):
    with torch.no_grad():
        query = model(query_img.cuda())
    erased_heatmap = generate_heatmap(model, query_img)
    index = get_index(gallery_feature, gallery_cam, gallery_label, query)

    try:  # Visualize Ranking Result
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 14, 1)
        imshow(query_path, 'query', query_vis)
        ax.axis('off')
        ax = plt.subplot(1, 14, 2)
        ax.axis('off')
        imshow(query_path, 'hmap', heatmap)
        ax = plt.subplot(1, 14, 3)
        ax.axis('off')
        imshow(query_path, 'hmap_erased', erased_heatmap)

        for i in range(10):
            ax = plt.subplot(1, 14, i + 4)
            ax.axis('off')
            img_path, _ = image_datasets['gallery'].imgs[index[i]]
            label = gallery_label[index[i]]
            imshow(img_path)
            if label == query_label:
                ax.set_title('%d' % (i + 1), color='green')
            else:
                ax.set_title('%d' % (i + 1), color='red')
    except RuntimeError:
        for i in range(10):
            img_path = image_datasets.imgs[index[i]]
            print(img_path[0])
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
    directory = './network_attention/' + str(j) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory + str(img_num).zfill(3) + ".png")


def save_fig(query, size, i):
    fig = plt.figure()
    img = query.permute(1, 2, 0)
    plt.imshow(img)
    fig.savefig('./network_attention/network_attention_' + str(size) + '_' + str(i).zfill(3) + '.png')


def random_erase(query, vis):
    sizes = [32, 64, 128]
    aug_query = []
    i=0
    for size in sizes:
        row, col = size, size
        start, end = 0, 0
        while start + row <= 256:
            end = 0
            while end + col <= 128:
                temp = query.clone()
                temp[0, start:start + row, end:end + col] = 0
                temp[1, start:start + row, end:end + col] = 0
                temp[2, start:start + row, end:end + col] = 0

                end = end + 5
                aug_query.append(temp)
                i += 1
                # if not vis:
                #   save_fig(temp, size, i)
            start = start + 5

    return aug_query


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

gallery_feature, gallery_cam, gallery_label, query_feature, query_cam, query_label = load_features()

query_label = query_label[qi]

query_img = Image.open(query_path)
query = data_transforms(query_img)
query_vis = data_transforms_visualize(query_img)

temp = query.clone()
heatmap = generate_heatmap(model, temp.unsqueeze_(0))

query = random_erase(query, False)
query_vis = random_erase(query_vis, True)
print("New Images generated: ", len(query))
print('Searching for top 10 nearest images to the given query image with different Random Erasing patches: ')
model = get_model()


for i, aug_img in enumerate(query):
    aug_img.unsqueeze_(0)
    evaluate(model, heatmap, aug_img, query_vis[i], qi, i)
print('Finished inferencing. Results are saved in ./network_attention folder.')
