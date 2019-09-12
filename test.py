# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, PCB_test
from train_and_test_siamese import *

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='59', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='../Datasets/VeRi_with_plate/pytorch', type=str, help='./test_data')
parser.add_argument('--name', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--parts', default=6, type=int, help='batchsize')
parser.add_argument('--PCB_Ver', default=1, type=int, help='Divide feature maps horizontally or vertically (1 or 0)')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--RPP', action='store_true', help='use refined part pooling in PCB or not')
parser.add_argument('--use_siamese', action='store_true', help='use siamese')
parser.add_argument('--use_ftnet', action='store_true', help='use siamese')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')
parser.add_argument('--CB', action='store_true', help='use checkerboard partitioning or not.')
parser.add_argument('--mixed', action='store_true', help='use mixed partitioning or not.')
parser.add_argument('--share_conv', action='store_true', help='use 1*1 conv in PCB or not')

opt = parser.parse_args()
### load config ###
# load the training config

if opt.use_ftnet is False and opt.use_siamese is False and opt.PCB is False:
    print("No model selected. Please select at least one model: use_ftnet or use_siamese")
    exit()

if opt.use_ftnet:
    name = "ft_ResNet"
elif opt.use_siamese:
    name = "siamese"
elif opt.PCB:
    name = "ft_ResNet_PCB"

opt.nclasses = 575

h, w = 256, 128
if opt.PCB and opt.PCB_Ver:
    h, w = 384, 192
elif opt.PCB and not opt.PCB_Ver:
    h, w = 192, 384

config_path = os.path.join('./model', name, 'opts.yaml')
if os.path.isfile(config_path):
    with open(config_path, 'r') as stream:
            config = yaml.load(stream)
    opt.fp16 = config['fp16']
    opt.PCB = config['PCB']
    opt.use_dense = config['use_dense']
    opt.use_ftnet = config['use_ftnet']
    opt.use_NAS = config['use_NAS']
    opt.stride = config['stride']
    opt.use_siamese = config['use_siamese']

    if 'nclasses' in config:  # tp compatible with old config files
        opt.nclasses = config['nclasses']
    else:
        opt.nclasses = 575

print("Model name: ", name)
print("Epoch: ", opt.which_epoch)
print("Use_siamese: ", opt.use_siamese)
print("Use_ftnet: ", opt.use_ftnet)
print("nclasses: ", opt.nclasses)
print("Use_PCB: ", opt.PCB)



str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
if opt.use_siamese:
    trans = [
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ]
else:
    trans = [
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ############### Ten Crop
        #transforms.TenCrop(224),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.ToTensor()(crop)
          #      for crop in crops]
           # )),
        #transforms.Lambda(lambda crops: torch.stack(
         #   [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop)
          #       for crop in crops]
          # ))
    ]
data_transforms = transforms.Compose(trans)

print(data_transforms)
data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()


######################################################################
# Load model
#---------------------------
def load_network(network):
    modelname = 'net'
    if opt.PCB and opt.RPP:
        save_path = os.path.join('./model', name, modelname + '_full_%03d.pth'%int(opt.which_epoch))
    else:
        save_path = os.path.join('./model', name, modelname + '_%03d.pth'%int(opt.which_epoch))
    print('loading model from: ', save_path)

    network.load_state_dict(torch.load(save_path))
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


def get_features(model, img, label):
    n, c, h, w = img.size()
    ff = torch.FloatTensor(n,512).zero_()

    if opt.PCB:
        ff = torch.FloatTensor(n,2048,opt.parts).zero_() # we have six parts

    for i in range(2):
        if(i==1):
            img = fliplr(img)
        input_img = Variable(img.cuda())
        #if opt.fp16:
        #    input_img = input_img.half()
        outputs = model(input_img) 
        f = outputs.data.cpu().float()
        ff = ff+f
    # norm feature
    if opt.PCB:
        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.parts) 
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

    return ff
    

def extract_feature(model, dataloaders):
    features_final = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)

        features = get_features(model, img, label)
        #print(features.shape)

        if opt.mixed:
            """
            model.parts_ver=0
            model.avgpool=nn.AdaptiveAvgPool2d((1, opt.parts))
            features_hor = get_features(model, img, label)
            features = torch.add(features, features_hor)
            """
            model.checkerboard=True
            model.avgpool=nn.AdaptiveAvgPool2d((int(opt.parts / 2), 2))
            features_cb = get_features(model, img, label)

            features = torch.add(features, features_cb)
            features = torch.div(features, 2)
        features_final = torch.cat((features_final,features), 0)

        print(features_final.shape)
    return features_final


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        # replace camera[0] with camera[0:3] for VeRi dataset
        camera_id.append(int(camera[0:3]))
    return camera_id, labels


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)


if opt.use_siamese:
    get_siamese_features(gallery_cam, gallery_label, query_cam, query_label, opt.nclasses)


if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses)
elif opt.use_ftnet:
    model_structure = ft_net(opt.nclasses, stride=opt.stride)


if opt.PCB:
    model_structure = PCB(opt.nclasses, num_bottleneck=256, num_parts=opt.parts, parts_ver=opt.PCB_Ver, 
                          checkerboard=opt.CB, share_conv=opt.share_conv)
if opt.RPP:
    model_structure = model_structure.convert_to_rpp()
    
#if opt.fp16:
#    model_structure = network_to_half(model_structure)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if opt.PCB:
    #if opt.fp16:
    #    model = PCB_test(model[1])
    #else:
        model = PCB_test(model, num_parts=opt.parts, parts_ver=opt.PCB_Ver, checkerboard=opt.CB)

else:
    #if opt.fp16:
        #model[1].model.fc = nn.Sequential()
        #model[1].classifier = nn.Sequential()
    #else:
        model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders['gallery'])
    query_feature = extract_feature(model, dataloaders['query'])
    if opt.multi:
        mquery_feature = extract_feature(model, dataloaders['multi-query'])
    
# Save to Matlab for check
result = {'gallery_f': gallery_feature.numpy(), 'gallery_label': gallery_label, 'gallery_cam': gallery_cam,
          'query_f': query_feature.numpy(), 'query_label': query_label, 'query_cam': query_cam}
scipy.io.savemat('./model/' + name + '/pytorch_result_VeRi.mat', result)
if opt.multi:
    result = {'mquery_f': mquery_feature.numpy(), 'mquery_label': mquery_label, 'mquery_cam': mquery_cam}
    scipy.io.savemat('./model/' + name + '/multi_query.mat', result)
