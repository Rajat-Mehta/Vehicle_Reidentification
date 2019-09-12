# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import ft_net, ft_net_dense, PCB, PCB_test
import yaml

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='69', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='../Datasets/VeRi_with_plate/pytorch', type=str, help='./test_data')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--fusion', action='store_true', help='use fusion')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')
parser.add_argument('--PCB_H', action='store_true', help='Use PCB_Horizontal model in fusion')
parser.add_argument('--PCB_V', action='store_true', help='Use PCB_Vertical model in fusion')
parser.add_argument('--PCB_CB', action='store_true', help='Use PCB_CheckerBoard model in fusion')
parser.add_argument('--parts', default=6, type=int, help='number of parts in PCB')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
if opt.PCB:
    name = "siamese_PCB"
elif opt.fusion:
    name = "fusion"
    fusion_supp_model = "ft_ResNet_PCB"
    fusion_supp_which_epoch = 59
else:
    name = "siamese"
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

config_path = os.path.join('./model', name, 'opts.yaml')
if os.path.isfile(config_path):
    with open(config_path, 'r') as stream:
            config = yaml.load(stream)
    opt.fp16 = config['fp16']
    opt.PCB = config['PCB']
    opt.use_dense = config['use_dense']

    if 'nclasses' in config:  # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else:
        opt.nclasses = 575

print("Model name: ", name)
print("Epoch: ", opt.which_epoch)
print("nclasses: ", opt.nclasses)
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
h, w = 384,192
data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery', 'query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
def load_network(network, network_supp=None, fusion_support=False):

    save_filename = 'net_%03d.pth' % int(opt.which_epoch)
    save_path = os.path.join('./model', name, save_filename)
    print('Loading siamese model from: ', save_path)
    network.load_state_dict(torch.load(save_path))
    
    if fusion_support:
        save_filename_supp = 'net_%03d.pth' % int(fusion_supp_which_epoch)
        save_path_supp = os.path.join('./model', fusion_supp_model, save_filename_supp)
        print('Loading fusion support model from: ', save_path_supp)
        network_supp.load_state_dict(torch.load(save_path_supp))
        return network, network_supp
    
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_feature(model, dataloaders, model_list=None):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, 2048).zero_()
        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_() # we have six parts
        if opt.fusion:
            ff_PCB = torch.FloatTensor(n,2048,6).zero_() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            f = model(input_img)
            f = f.data.cpu().float()
            ff = ff+f
            
            if opt.fusion:
                temp = torch.FloatTensor(n,2048,6).zero_()
                for model_pcb in model_list:
                    f_PCB = model_pcb(input_img)
                    f_PCB = f_PCB.data.cpu().float()
                    temp = temp+f_PCB
                temp = temp/len(model_list)
                ff_PCB = ff_PCB+temp
                
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            
            if opt.fusion:
                fnorm_PCB = torch.norm(ff_PCB, p=2, dim=1, keepdim=True) * np.sqrt(6) 
                ff_PCB = ff_PCB.div(fnorm_PCB.expand_as(ff_PCB))
                ff_PCB = ff_PCB.view(ff_PCB.size(0), -1)
                ff = torch.cat((ff,ff_PCB),1)

        features = torch.cat((features,ff), 0)
    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]

        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0:3]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs
# mquery_path = image_datasets['multi-query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

# mquery_cam,mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
else:
    model_structure = ft_net(opt.nclasses, return_f=True, num_bottleneck=2048)

if opt.PCB:
    model_structure = PCB(opt.nclasses, return_f=True, num_bottleneck=512)


def load_network_PCB(network, name):
    if name == "PCB_V":
        save_path = os.path.join('./model', 'ft_ResNet_PCB', 'vertical/part6_vertical', 'net_%03d.pth'%59)
    elif name == "PCB_H":
        save_path = os.path.join('./model', 'ft_ResNet_PCB', 'horizontal/part6_horizontal', 'net_%03d.pth'%49)
    elif name == "PCB_CB":
        save_path = os.path.join('./model', 'ft_ResNet_PCB', 'checkerboard/part6_CB/with_erasing', 'net_%03d.pth'%59)

    print('PCB_ResNet: Loading pretrainded model from: ', save_path)
    network.load_state_dict(torch.load(save_path))
    return network


model_list = []

if opt.PCB_H:
    model_structure_PCB = PCB(575, num_bottleneck=256, num_parts=opt.parts, parts_ver=0)
    model_PCB_H = load_network_PCB(model_structure_PCB,'PCB_H')
    model_PCB_H = PCB_test(model_PCB_H, num_parts=opt.parts)
    model_PCB_H = model_PCB_H.eval()
    if use_gpu:
        model_PCB_H = model_PCB_H.cuda()
    print("PCB_Horizontal model", model_PCB_H)
    model_list.append(model_PCB_H)

if opt.PCB_V:
    model_structure_PCB = PCB(575, num_bottleneck=256, num_parts=opt.parts, parts_ver=1)
    model_PCB_V = load_network_PCB(model_structure_PCB,'PCB_V')
    model_PCB_V = PCB_test(model_PCB_V, num_parts=opt.parts)
    model_PCB_V = model_PCB_V.eval()
    if use_gpu:
        model_PCB_V = model_PCB_V.cuda()
    print("PCB_Vertical model", model_PCB_V)
    model_list.append(model_PCB_V)
if opt.PCB_CB:
    model_structure_PCB = PCB(575, num_bottleneck=256, num_parts=opt.parts, parts_ver=1, checkerboard=True)
    model_PCB_CB = load_network_PCB(model_structure_PCB,'PCB_CB')
    model_PCB_CB = PCB_test(model_PCB_CB, num_parts=opt.parts)
    model_PCB_CB = model_PCB_CB.eval()
    if use_gpu:
        model_PCB_CB = model_PCB_CB.cuda()
    print("PCB_Checkerboard model", model_PCB_CB)
    model_list.append(model_PCB_CB)

if opt.fusion:
    """ load ResNet_PCB model here"""
    model = load_network(model_structure)
    model = model.eval()

else:
    model = load_network(model_structure)
    model = model.eval()

if opt.PCB:
    model = PCB_test(model)

#model = model_structure

# Remove the final fc layer and classifier layer
#if not opt.PCB:
#    model.model.fc = nn.Sequential()
#    model.classifier = nn.Sequential()
#else:
#    model = PCB_test(model)

# Change to test mode
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature = extract_feature(model, dataloaders['gallery'], model_list)
query_feature = extract_feature(model, dataloaders['query'], model_list)
if opt.multi:
    mquery_feature = extract_feature(model, dataloaders['multi-query'])
    
# Save to Matlab for check
result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
scipy.io.savemat('./model/' + name + '/pytorch_result_VeRi.mat', result)
'''
if opt.multi:
    result = {'mquery_f':mquery_feature.numpy(),'mquery_label':mquery_label,'mquery_cam':mquery_cam}
    scipy.io.savemat('../model/' + name + '/multi_query.mat', result)

'''
