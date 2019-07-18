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

erase_size = [32, 64, 80, 90, 128]
parser_na = argparse.ArgumentParser(description='Visualizing')

parser_na.add_argument('--data_dir', default='../Datasets/VeRi_with_plate/pytorch', type=str, help='./test_data')
parser_na.add_argument('--name', default='ft_ResNet', type=str, help='save model path')
parser_na.add_argument('--batchsize', default=512, type=int, help='batchsize')
parser_na.add_argument('--query_index', default=800, type=int, help='test_image_index')
parser_na.add_argument('--acc', action='store_true', help='heatmap')


opts = parser_na.parse_args()

qi = opts.query_index

config_path = os.path.join('./model', opts.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opts.fp16 = config['fp16']
opts.PCB = config['PCB']
opts.use_dense = config['use_dense']
opts.use_NAS = config['use_NAS']
opts.stride = config['stride']

opts.h = 256
opts.w = 128
opts.acc_hm = True
if 'nclasses' in config:  # tp compatible with old config files
    opts.nclasses = config['nclasses']
else:
    opts.nclasses = 751


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


def get_model():
    model = ft_net(class_num=575)
    model.load_state_dict(torch.load('./model/ft_ResNet/net_99.pth'))
    model.classifier.classifier = nn.Sequential()
    model = model.eval().cuda()

    return model


def load_features():
    result = scipy.io.loadmat('./model/' + opts.name + '/pytorch_result_VeRi_99.mat')
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


def imshow(path, title=None, query_vis=None):
    """Imshow for Tensor."""
    # im = plt.imread(path)

    im = Image.open(path)
    im = data_transforms_visualize(im)
    im = im.permute(1, 2, 0)

    if title == 'hmap_orig' or title == 'hmap_erased' or ('hmap' in title):
        # plt.clim(0,1)
        heatmap = plt.imshow(query_vis, cmap='jet')
        # cb = plt.colorbar(heatmap, orientation='vertical',  pad=0.4)
        # cb.set_ticks([])
    elif query_vis is not None:
        query_vis = query_vis.permute(1, 2, 0)
        plt.imshow(query_vis)

    else:
        plt.imshow(im)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def plot_row(index, heatmap, query_vis, query_orig, heatmap_orig):
    ax = plt.subplot(3, 14, 1)
    imshow(query_path, 'query_orig', query_orig)
    ax.axis('off')

    ax = plt.subplot(3, 14, 2)
    ax.axis('off')
    imshow(query_path, 'hmap_orig', heatmap_orig)

    ax = plt.subplot(3, 14, 3)
    imshow(query_path, 'query_erased', query_vis)
    ax.axis('off')

    ax = plt.subplot(3, 14, 4)
    ax.axis('off')
    imshow(query_path, 'hmap_erased', heatmap)

    for i in range(10):
        ax = plt.subplot(3, 14, i+5)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]

        label = gallery_label[index[i]]
        imshow(img_path, 'result')
        if label == query_label:
            ax.set_title('%d' % (i + 1), color='green')
        else:
            ax.set_title('%d' % (i + 1), color='red')

        result_img = Image.open(img_path)
        result_img = data_transforms(result_img)
        result_heatmap = generate_heatmap(model, result_img.unsqueeze_(0))

        ax = plt.subplot(3, 14, i + 19)
        ax.axis('off')
        imshow(query_path, 'hmap_' + str(i+1), result_heatmap)


def evaluate(model, query_img, query_vis, j, img_num, query_orig, heatmap_orig):
    with torch.no_grad():
        query = model(query_img.cuda())

    erased_heatmap = generate_heatmap(model, query_img)

    index = get_index(gallery_feature, gallery_cam, gallery_label, query)

    try:  # Visualize Ranking Result
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(20, 8))
        plt.subplots_adjust(left=0.08, right=0.9, bottom=0.1, top=0.85)

        plot_row(index, erased_heatmap, query_vis, query_orig, heatmap_orig)

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


def random_erase(query, sizes):
    aug_query = []
    i=0
    for size in sizes:
        row, col = size, size
        start, end = 0, 0
        r = 0
        while start + row <= 256:
            end = 0
            c = 0
            while end + col <= 128:
                temp = query.clone()
                temp[0, start:start + row, end:end + col] = 0
                temp[1, start:start + row, end:end + col] = 0
                temp[2, start:start + row, end:end + col] = 0

                end = end + 8
                aug_query.append(temp)
                c += 1
                i += 1
                # if not vis:
                #   save_fig(temp, size, i)
            start = start + 8
            r += 1
    return aug_query, (r, c)


data_transforms = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data_transforms_visualize = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor()
])

image_datasets = {x: datasets.ImageFolder(os.path.join(opts.data_dir, x), data_transforms) for x in ['gallery', 'query']}

query_path, _ = image_datasets['query'].imgs[qi]

gallery_feature, gallery_cam, gallery_label, query_feature, query_cam, query_label = load_features()

query_label = query_label[qi]

query_img = Image.open(query_path)
query = data_transforms(query_img)
query_vis = data_transforms_visualize(query_img)

model = get_model()

temp = query.clone()
heatmap = generate_heatmap(model, temp.unsqueeze_(0))

query_orig = query
query_orig_vis = query_vis


if opts.acc:
    print("Printing accuracy heatmap")
    res_acc = []
    fig = plt.figure(figsize=(10, 4))
    plot = 0
    for i, size in enumerate(erase_size):
        temp = query.clone()
        temp, op_size = random_erase(temp, [size])
        print("New Images generated for size", size, ": ", len(temp))
        acc = []
        for j, aug_img in enumerate(temp):
            count = 0
            aug_img.unsqueeze_(0)
            with torch.no_grad():
                query_vec = model(aug_img.cuda())
            index = get_index(gallery_feature, gallery_cam, gallery_label, query_vec)
            for i in range(10):
                label = gallery_label[index[i]]
                if label == query_label:
                    count += 1
            acc.append(1 - (count / 10))
        plot += 1
        acc = np.reshape(acc, op_size)
        ax = plt.subplot(1, 7, 1)
        plt.subplots_adjust(left=0.08, right=0.9, bottom=0.1, top=0.85)
        imshow(query_path, 'query_orig', query_orig_vis)
        ax.axis('off')

        ax = plt.subplot(1, 7, 2)
        ax.axis('off')
        imshow(query_path, 'hmap_orig', heatmap)

        ax = plt.subplot(1, 7, plot+2)
        ax.axis('off')
        imshow(query_path, 'hmap_'+str(size)+'*'+str(size), acc)
        res_acc.append(acc)
    directory = './network_attention/' + str(opts.query_index)
    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(directory + '/acc_heatmap.png')


query, _ = random_erase(query, erase_size)
query_vis, _ = random_erase(query_vis, erase_size)

print("New Images generated: ", len(query))
print('Searching for top 10 nearest images to the given query image with different Random Erasing patches: ')

for i, aug_img in enumerate(query):
    aug_img.unsqueeze_(0)
    evaluate(model, aug_img, query_vis[i], qi, i, query_orig_vis, heatmap)
print('Finished inferencing. Results are saved in ./network_attention folder.')
