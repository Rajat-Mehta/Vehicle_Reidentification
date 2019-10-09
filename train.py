# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
from model import ft_net, ft_net_dense, ft_net_NAS, PCB, auto_encoder, PCB_test
from augmentation import RandomErasing
from augmentation import ImgAugTransform

import yaml
import math
from shutil import copyfile
from train_and_test_siamese import *
from utils import update_average, get_model_list, load_network, save_network, make_weights_for_balanced_classes
import pickle
from kmeans_pytorch.kmeans import lloyd
import torch.utils.data as utils
from pykeops.torch import LazyTensor
from torchvision.utils import save_image

use_cuda = torch.cuda.is_available()
dtype = 'float32' if use_cuda else 'float64'
torchtype = {'float32': torch.float32, 'float64': torch.float64}

version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--data_dir',default='../Datasets/VeRi_with_plate/pytorch',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--epochs', default=60, type=int, help='epochs')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--flip', default=0, type=float, help='Horizontal flip probability, in [0,1]')
parser.add_argument('--rotate', default=0, type=float, help='Rotate images')
parser.add_argument('--translate', nargs='+', type=float, help='Translate images, give values between 0 and 1')
parser.add_argument('--aug_comb', action='store_true', help='Use a combination of augmentations')
parser.add_argument('--scale', nargs='+', type=float, help='Scale images(zoom in or zoom out), give values between 0 and 2')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_ftnet', action='store_true', help='use ftnet')
parser.add_argument('--use_siamese', action='store_true', help='use siamese')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50')
parser.add_argument('--RPP', action='store_true', help='use refined part pooling in PCB or not')
parser.add_argument('--cluster', action='store_true', help='use k means clustering to partition feature maps in PCB')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--resume', action='store_true', help='resume training')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=128, type=int, help='width')
parser.add_argument('--pool', default='avg', type=str, help='pool avg')
parser.add_argument('--parts', default=6, type=int, help='number of parts in PCB')
parser.add_argument('--PCB_Ver', default=1, type=int, help='Divide feature maps vertically or horizontally (1 or 0)')
parser.add_argument('--CB', action='store_true', help='use checkerboard partitioning in PCB')
parser.add_argument('--no_induction', action='store_true', help='dont use induced training in PCB')
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory')
parser.add_argument('--mixed_part', action='store_true', help='use mixed partitioning training: first vertical, then horizontal and then checkerboard')
parser.add_argument('--share_conv', action='store_true', help='use 1*1 conv in PCB or not')
parser.add_argument('--re_compute_features', action='store_true', help='re compute train set features for KMeans clustering model or not?')
parser.add_argument('--auto_encode', action='store_true', help='train auto encoder for dimensionality reduction')
parser.add_argument('--retrain_autoencode', action='store_true', help='retrain auto encoder for dimensionality reduction')
parser.add_argument('--veri_wild', action='store_true', help='use veri wild dataset')

opt = parser.parse_args()

fp16 = opt.fp16
data_dir = opt.data_dir
if opt.veri_wild:
    data_dir = '../Datasets/VeRI-Wild/pytorch'
    opt.data_dir = '../Datasets/VeRI-Wild/pytorch'
if opt.use_dense is False and opt.use_siamese is False and opt.use_NAS is False and opt.use_ftnet is False and opt.PCB is False:
    print("No model selected. Please select at least one model to train like: use_ftnet or use_siamese")
    exit()


if opt.use_dense:
    name = "ft_net_dense"
elif opt.use_siamese:
    name = "siamese"
elif opt.use_NAS:
    name = "ft_net_NAS"
elif opt.PCB:
    name = "ft_ResNet_PCB"
elif opt.use_ftnet:
    name = "ft_ResNet"

opt.name = name

if opt.PCB and opt.PCB_Ver:
    opt.h, opt.w = 384, 192
elif opt.PCB and not opt.PCB_Ver:
    opt.h, opt.w = 192, 384


if opt.resume:
    print("Resuming training from presaved model. ")
    model, opt, start_epoch = load_network(name, opt)
    start_epoch += 1
else:
    start_epoch = 0

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids)>0:
    print("GPU number", gpu_ids)
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#
transform_train_list = []

# transform_train_list = transform_train_list + [# transforms.RandomResizedCrop(size=128, scale=(0.75,1.0),
# ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)]
transform_train_list = transform_train_list + [transforms.Resize((opt.h, opt.w), interpolation=3)]
# transform_train_list = transform_train_list + [transforms.Pad(10)]
# transform_train_list = transform_train_list + [transforms.RandomCrop((opt.h, opt.w))]

if opt.flip:
    transform_train_list = transform_train_list + [transforms.RandomHorizontalFlip(p=opt.flip)]


if opt.aug_comb:
    transform_train_list = transform_train_list + [ImgAugTransform(rt=opt.rotate, tl=opt.translate, scale=opt.scale,
                                                                   aug_comb=opt.aug_comb)]

else:
    if opt.rotate:
        transform_train_list = transform_train_list + [ImgAugTransform(rt=opt.rotate)]

    if opt.translate:
        transform_train_list = transform_train_list + [ImgAugTransform(tl=opt.translate)]

    if opt.scale:
        transform_train_list = transform_train_list + [ImgAugTransform(scale=opt.scale)]

transform_train_list = transform_train_list + [transforms.ToTensor()]
transform_train_list = transform_train_list + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]


transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w), interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.PCB:
    transform_train_list = []
    transform_train_list = transform_train_list + [
            transforms.Resize((opt.h, opt.w), interpolation=3)
            ] 

    if opt.re_compute_features:
        
        transform_train_list = transform_train_list + [transforms.ToTensor()]
        transform_train_list = transform_train_list + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    else:
        transform_train_list = transform_train_list + [transforms.RandomHorizontalFlip()]
        
        if opt.aug_comb:
            transform_train_list = transform_train_list + [ImgAugTransform(rt=opt.rotate, tl=opt.translate, scale=opt.scale, aug_comb=opt.aug_comb)]
        else:
            if opt.rotate:
                transform_train_list = transform_train_list + [ImgAugTransform(rt=opt.rotate)]

            if opt.translate:
                transform_train_list = transform_train_list + [ImgAugTransform(tl=opt.translate)]

            if opt.scale:
                transform_train_list = transform_train_list + [ImgAugTransform(scale=opt.scale)]
            
        transform_train_list = transform_train_list + [transforms.ToTensor()]
        transform_train_list = transform_train_list + [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    transform_val_list = [
        transforms.Resize(size=(opt.h, opt.w), interpolation=3),  # Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if not opt.re_compute_features and opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if not opt.re_compute_features and opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + \
                           transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'val': transforms.Compose(transform_val_list),
}


def visualize_dataset_new(dataloaders):
    """Imshow for Tensor."""

    fig = plt.figure(figsize=(9, 13))
    columns = 4
    rows = 5
    input, label = next(iter(dataloaders['train']))

    fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.5, hspace=0.5)
    # prep (x,y) for extra plotting
    xs = np.linspace(0, 2 * np.pi, 60)  # from 0 to 2pi
    ys = np.abs(np.sin(xs))  # absolute of sine

    # ax enables access to manipulate each of subplots
    ax = []
    j=0

    for i in range(columns * rows):
        img = input[j]
        lbl = label[j].item()
        img = img.permute(1, 2, 0)
        img = img.numpy()
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title(str(i))  # set title
        plt.imshow(img)
        j += 1

    # do extra plots on selected axes/subplots
    # note: index starts with 0
    ax[2].plot(xs, 3 * ys)
    ax[19].plot(ys ** 2, xs)

    plt.savefig(os.path.join("./model", name, "dataset.png"))


def visualize_dataset(dataloaders):
    """Imshow for Tensor."""

    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    j = 0
    for i in range(1, columns * rows + 1):
        for batch_idx, (inputs, labels) in enumerate(dataloaders['train']):
            inputs = inputs
            break

        img = inputs[j]
        label = labels[j]
        img = img.permute(1, 2, 0)
        img = img.numpy()
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        j+=1
    plt.show()
    plt.savefig("./model/ft_ResNet/ft_ResNet_HFlip/dataset.png")


train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                          data_transforms['val'])

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
              for x in ['train', 'val']}

#visualize_dataset_new(dataloaders)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

since = time.time()
inputs, classes = next(iter(dataloaders['train']))
print(time.time()-since)
######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, scheduler, stage=None, num_epochs=25):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['train'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch
    for epoch in range(num_epochs - start_epoch):
        epoch = epoch + start_epoch
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                if not opt.PCB:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                else:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = opt.parts
                    for i in range(num_part):
                        part[i] = outputs[i]
                    if num_part == 6:
                        score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5])
                    elif num_part == 8:
                        score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3]) + sm(part[4]) + sm(part[5]) + sm(part[6]) + sm(part[7])
                    else:
                        score = sm(part[0]) + sm(part[1]) + sm(part[2]) + sm(part[3])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part - 1):
                        loss += criterion(part[i + 1], labels)

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if fp16:  # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    ##########

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()
                if epoch % 10 == 9 or ((stage =='full' or stage =='rpp') and epoch % 10 == 4):
                    save_network(model, epoch, stage)
                draw_curve(epoch, stage)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(last_model_wts)
    save_network(model, 'last', stage)

    return model


######################################################################
# Draw Curve
# ---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")


def draw_curve(current_epoch, stage=None):
    if stage:
        filename='train_' + stage + '.jpg'
    else:
        filename='train.jpg'    
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    
    fig.savefig( os.path.join('./model',name, filename))

######################################################################
# Save model
# ---------------------------


def save_network(network, epoch_label, stage=None):
    if opt.RPP:
        netname='net_' + stage
    else:
        netname='net'

    if isinstance(epoch_label, int):
        save_filename = netname + '_%03d.pth'% epoch_label
    else:
        save_filename = netname + '_%s.pth'% epoch_label

    save_path = os.path.join('./model', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


def save_train_config_files():
    dir_name = os.path.join('./model', name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    # record every run
    copyfile('./train.py', dir_name + '/train.py')
    copyfile('./model.py', dir_name + '/model.py')
    copyfile('./augmentation.py', dir_name + '/augmentation.py')

    if opt.use_siamese:
        copyfile('./train_and_test_siamese.py', dir_name + '/train_and_test_siamese.py')

    # save opts
    with open('%s/opts.yaml' % dir_name, 'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)


def pcb_train(model, criterion, stage, num_epoch):
    ignored_params = list(map(id, model.model.fc.parameters()))
    ignored_params += (list(map(id, model.classifier0.parameters()))
                     + list(map(id, model.classifier1.parameters()))
                     + list(map(id, model.classifier2.parameters()))
                     + list(map(id, model.classifier3.parameters()))
                     )
    if opt.parts == 6:
        ignored_params+= (list(map(id, model.classifier4.parameters()))
                          + list(map(id, model.classifier5.parameters()))
                          )
    if opt.parts == 8:
        ignored_params+= (list(map(id, model.classifier4.parameters()))
                          + list(map(id, model.classifier5.parameters()))
                          + list(map(id, model.classifier6.parameters()))
                          + list(map(id, model.classifier7.parameters()))
                          )

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    if opt.parts == 6:
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1*opt.lr},
            {'params': model.model.fc.parameters(), 'lr': opt.lr},
            {'params': model.classifier0.parameters(), 'lr': opt.lr},
            {'params': model.classifier1.parameters(), 'lr': opt.lr},
            {'params': model.classifier2.parameters(), 'lr': opt.lr},
            {'params': model.classifier3.parameters(), 'lr': opt.lr},
            {'params': model.classifier4.parameters(), 'lr': opt.lr},
            {'params': model.classifier5.parameters(), 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif opt.parts == 8:
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1*opt.lr},
            {'params': model.model.fc.parameters(), 'lr': opt.lr},
            {'params': model.classifier0.parameters(), 'lr': opt.lr},
            {'params': model.classifier1.parameters(), 'lr': opt.lr},
            {'params': model.classifier2.parameters(), 'lr': opt.lr},
            {'params': model.classifier3.parameters(), 'lr': opt.lr},
            {'params': model.classifier4.parameters(), 'lr': opt.lr},
            {'params': model.classifier5.parameters(), 'lr': opt.lr},
            {'params': model.classifier6.parameters(), 'lr': opt.lr},
            {'params': model.classifier7.parameters(), 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1*opt.lr},
            {'params': model.model.fc.parameters(), 'lr': opt.lr},
            {'params': model.classifier0.parameters(), 'lr': opt.lr},
            {'params': model.classifier1.parameters(), 'lr': opt.lr},
            {'params': model.classifier2.parameters(), 'lr': opt.lr},
            {'params': model.classifier3.parameters(), 'lr': opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    if fp16:
        # model = network_to_half(model)
        # optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
        model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")
    
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        stage=stage, num_epochs=num_epoch)
    return model


######################################################################
# RPP train
# ------------------
# Setp 2&3: train the rpp layers
# According to original paper, we set the learning rate at 0.01 for rpp layers.

def rpp_train(model, criterion, stage, num_epoch):
    if opt.no_induction:
        ignored_params = list(map(id, model.model.fc.parameters()))
        ignored_params += (list(map(id, model.avgpool.parameters()))
                        + list(map(id, model.classifier0.parameters()))
                        + list(map(id, model.classifier1.parameters()))
                        + list(map(id, model.classifier2.parameters()))
                        + list(map(id, model.classifier3.parameters()))
                        )
        if opt.parts == 6:
            ignored_params+= (list(map(id, model.classifier4.parameters()))
                            + list(map(id, model.classifier5.parameters()))
                            )
        elif opt.parts == 8:
            ignored_params+= (list(map(id, model.classifier4.parameters()))
                            + list(map(id, model.classifier5.parameters()))
                            + list(map(id, model.classifier6.parameters()))
                            + list(map(id, model.classifier7.parameters()))
                            )

        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        if opt.parts == 6:
            optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': model.avgpool.parameters(), 'lr': opt.lr},
                {'params': model.model.fc.parameters(), 'lr': opt.lr},
                {'params': model.classifier0.parameters(), 'lr': opt.lr},
                {'params': model.classifier1.parameters(), 'lr': opt.lr},
                {'params': model.classifier2.parameters(), 'lr': opt.lr},
                {'params': model.classifier3.parameters(), 'lr': opt.lr},
                {'params': model.classifier4.parameters(), 'lr': opt.lr},
                {'params': model.classifier5.parameters(), 'lr': opt.lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        elif opt.parts == 8:
            optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': model.avgpool.parameters(), 'lr': opt.lr},
                {'params': model.model.fc.parameters(), 'lr': opt.lr},
                {'params': model.classifier0.parameters(), 'lr': opt.lr},
                {'params': model.classifier1.parameters(), 'lr': opt.lr},
                {'params': model.classifier2.parameters(), 'lr': opt.lr},
                {'params': model.classifier3.parameters(), 'lr': opt.lr},
                {'params': model.classifier4.parameters(), 'lr': opt.lr},
                {'params': model.classifier5.parameters(), 'lr': opt.lr},
                {'params': model.classifier6.parameters(), 'lr': opt.lr},
                {'params': model.classifier7.parameters(), 'lr': opt.lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        
        else:
            optimizer_ft = optim.SGD([
                {'params': base_params, 'lr': 0.1*opt.lr},
                {'params': model.avgpool.parameters(), 'lr': opt.lr},
                {'params': model.model.fc.parameters(), 'lr': opt.lr},
                {'params': model.classifier0.parameters(), 'lr': opt.lr},
                {'params': model.classifier1.parameters(), 'lr': opt.lr},
                {'params': model.classifier2.parameters(), 'lr': opt.lr},
                {'params': model.classifier3.parameters(), 'lr': opt.lr}
            ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        # Decay LR by a factor of 0.1 every 100 epochs (never use)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
        
    else:
        optimizer_ft = optim.SGD(model.avgpool.parameters(), lr=0.1*opt.lr,
                              weight_decay=5e-4, momentum=0.9, nesterov=True)

        # Decay LR by a factor of 0.1 every 100 epochs (never use)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        stage=stage, num_epochs=num_epoch)
    return model

######################################################################
# full train
# ------------------
# Step 4: train the whole net
# According to original paper, we set the difference learning rate for the whole net

def full_train(model, criterion, stage, num_epoch):
    ignored_params = list(map(id, model.model.fc.parameters()))
    ignored_params += (list(map(id, model.avgpool.parameters()))
                     + list(map(id, model.classifier0.parameters()))
                     + list(map(id, model.classifier1.parameters()))
                     + list(map(id, model.classifier2.parameters()))
                     + list(map(id, model.classifier3.parameters()))
                     )
    if opt.parts == 6:
        ignored_params+= (list(map(id, model.classifier4.parameters()))
                          + list(map(id, model.classifier5.parameters()))
                          )
    elif opt.parts == 8:
            ignored_params+= (list(map(id, model.classifier4.parameters()))
                            + list(map(id, model.classifier5.parameters()))
                            + list(map(id, model.classifier6.parameters()))
                            + list(map(id, model.classifier7.parameters()))
                            )

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    
    if opt.parts == 6:
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.01*opt.lr},
            {'params': model.avgpool.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.model.fc.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier0.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier1.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier2.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier3.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier4.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier5.parameters(), 'lr': 0.1*opt.lr},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    elif opt.parts == 8:
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.01*opt.lr},
            {'params': model.avgpool.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.model.fc.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier0.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier1.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier2.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier3.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier4.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier5.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier6.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier7.parameters(), 'lr': 0.1*opt.lr}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.01*opt.lr},
            {'params': model.avgpool.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.model.fc.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier0.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier1.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier2.parameters(), 'lr': 0.1*opt.lr},
            {'params': model.classifier3.parameters(), 'lr': 0.1*opt.lr},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    # Decay LR by a factor of 0.1 every 100 epochs (never use)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        stage=stage, num_epochs=num_epoch)
    return model


if not opt.resume:
    if opt.use_dense:
        model = ft_net_dense(len(class_names), opt.droprate)
    elif opt.use_NAS:
        model = ft_net_NAS(len(class_names), opt.droprate)
    else:
        model = ft_net(len(class_names), opt.droprate, opt.stride)



save_train_config_files()
criterion = nn.CrossEntropyLoss()

opt.nclasses = len(class_names)

if start_epoch >= 40:
    opt.lr = opt.lr * 0.1

if not opt.PCB:
    ignored_params = list(map(id, model.classifier.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.1 * opt.lr},
        {'params': model.classifier.parameters(), 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # Decay LR by a factor of 0.1 every 40 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)


def load_pcb(network):
    save_path = './model/ft_ResNet_PCB/vertical/part6_vertical/net_079.pth'
    network.load_state_dict(torch.load(save_path))
    network = network.eval().cuda()
    return network


def train_cluster(features, centers_path):
    print("Training KMeans clustering model for train data: ", features.shape)

    clusters_index, centers = KMeans(features, opt.parts)

    with open(centers_path, "wb") as c:
        pickle.dump(centers, c)
    print("Cluster centroids saved in pickle file.")

    return centers_path


def KMeans(x, K=10, Niter=10, verbose=True):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    start = time.time()
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):

        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    end = time.time()

    if verbose:
        print("K-means example with {:,} points in dimension {:,}, K = {:,}:".format(N, D, K))
        print('Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n'.format(
                Niter, end - start, Niter, (end-start) / Niter))

    return cl, c


def train_auto_encoder(auto_encode_model, model_PCB, criterion, optimizer):
    since = time.time()
    num_epochs = 60
    for epoch in range(num_epochs):
        for data in dataloaders['train']:
            # get the inputs
            inputs, labels = data
            
            now_batch_size, c, h, w = inputs.shape
            ff = torch.FloatTensor(now_batch_size,2048,opt.parts).zero_()
            if now_batch_size < opt.batchsize:  # skip the last batch
                continue
            if use_gpu:
                inputs = Variable(inputs.cuda().detach())
            else:
                inputs= Variable(inputs)

            ff = model_PCB(inputs)
            
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.parts) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            
            # ===================forward=====================
            output = auto_encode_model(ff)
            loss = criterion(output, ff)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        
        print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.item()))
        
        if epoch % 10 == 4 or epoch % 10 == 9:
            torch.save(auto_encode_model.state_dict(), './model/ft_ResNet_PCB/autoencoder/autoencoder_' + str(epoch) + '.pth')

    return auto_encode_model

if opt.auto_encode:
    model = PCB(len(class_names), num_bottleneck=256, num_parts=opt.parts, parts_ver=opt.PCB_Ver,
                        checkerboard=opt.CB, share_conv=opt.share_conv)
    model.load_state_dict(torch.load('./model/ft_ResNet_PCB/vertical/part6_vertical/net_079.pth'))
    model = PCB_test(model, num_parts=opt.parts, parts_ver=opt.PCB_Ver, checkerboard=opt.CB)
    model = model.cuda()

    if opt.retrain_autoencode:
        auto_enc_model = auto_encoder()
        auto_enc_model = auto_enc_model.cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(auto_enc_model.parameters(), lr=1e-3, weight_decay=1e-5)

        print("Autoencoder model structure: ")
        print(auto_enc_model)

        auto_enc_model = train_auto_encoder(auto_enc_model, model, criterion, optimizer)
    else:
        auto_enc_model = auto_encoder()
        auto_enc_model.load_state_dict(torch.load('./model/ft_ResNet_PCB/autoencoder/autoencoder_24.pth'))
        auto_enc_model = auto_enc_model.cuda()

   
if opt.PCB or opt.RPP :
    # step1: PCB training #
    if not opt.no_induction:
        print("STARTING STEP 1 OF PCB:")
        stage = 'pcb'

        if opt.mixed_part:
            print("Started Training PCB with checkerboard partitioning")
            model = PCB(len(class_names), num_bottleneck=256, num_parts=opt.parts, parts_ver=1, checkerboard=True, share_conv=opt.share_conv)
            model = model.cuda()
            print(model)
            model = pcb_train(model, criterion, stage, 40)
            print("Checkerboard partition training finished")

            print("Started Training PCB with vertical partitioning")
            model.checkerboard=False
            model.avgpool=nn.AdaptiveAvgPool2d((opt.parts, 1))
            print(model)
            start_epoch=40
            model = pcb_train(model, criterion, stage, 100)
            print("Vertical partition training finished")

        else:
            if opt.cluster:
                model = PCB(len(class_names), num_bottleneck=256, num_parts=opt.parts, parts_ver=opt.PCB_Ver,
                                    checkerboard=opt.CB, share_conv=opt.share_conv)
                print(model)
                model = load_pcb(model)
                centers_path= os.path.join('./model', name, 'centers.pkl')
                print("STARTING STEP 2 AND 3 OF PCB using KMeans Clustering:")

                # if opt.re_compute_features is false, make sure that cluster centers are already saved in centers.pkl file in the working directory
                if opt.re_compute_features:
                    print("Extracting feature vectors from train dataset, which will be used in KMeans clustering to generate opt.num_parts clusters")
                    features = torch.FloatTensor()
                    i=0
                    segment=0
                    #extracting feature vectors of random 3000 train images and saving in pickle files
                    for data in dataloaders['train']:    
                        if i % 50 == 49:
                            pickle_path = os.path.join('./model', name, 'train_features'+str(segment)+ '.pkl')
                            segment+=1
                            #pickle the list and save so that it can be reused later
                            with open(pickle_path, "wb") as f:
                                pickle.dump(features, f, protocol=4)
                            features = torch.FloatTensor()
                            break

                        i+=1
                        inputs, labels = data
                        inputs = Variable(inputs.cuda().detach())
                        labels = Variable(labels.cuda().detach())

                        x = model.model.conv1(inputs)
                        x = model.model.bn1(x)
                        x = model.model.relu(x)
                        x = model.model.maxpool(x)
                        x = model.model.layer1(x)
                        x = model.model.layer2(x)
                        x = model.model.layer3(x)
                        x = model.model.layer4(x)
                        x = x.view(x.shape[0], x.shape[1], -1)
                        x = x.permute(0, 2, 1)
                        x=x.data.cpu().float()
                        features = torch.cat((features,x),0)
                    
                    features = torch.FloatTensor()
                    
                    for i in range(segment):
                        pickle_path = os.path.join('./model', name, 'train_features'+str(i)+ '.pkl')
                        with open(pickle_path, "rb") as f:
                            temp = pickle.load(f)
                            features = torch.cat((features, temp))
                    features = features.view(-1, features.size(2))
                    print("Feature extraction finished")
                    print("Feeding extracted feature vectors to KMeans clustering model to generate clusters")
                    # start clustering with features vector
                    centers_path = train_cluster(features, centers_path)
                    print("Clusters generated")

                model = model.convert_to_rpp_cluster()
                
                model.avgpool.centers_path = centers_path
                model = model.cuda()
                print(model)
                # step4: whole net training #
                print("STARTING STEP 4 OF PCB:")
                stage = 'full'
                full_train(model, criterion, stage, opt.epochs)
            else:
                model = PCB(len(class_names), num_bottleneck=256, num_parts=opt.parts, parts_ver=opt.PCB_Ver,
                        checkerboard=opt.CB, share_conv=opt.share_conv)
                #model.load_state_dict(torch.load('./model/ft_ResNet_PCB/vertical/part6_vertical/net_079.pth'))
                model = model.cuda()
                print(model)
                model = pcb_train(model, criterion, stage, opt.epochs)

    # step2&3: RPP training #
    if opt.RPP:
        print("STARTING STEP 2 AND 3 OF PCB:")
        stage = 'rpp'
        epochs = 20
        if opt.no_induction:
            model = PCB(len(class_names), num_bottleneck=256, num_parts=opt.parts, parts_ver=opt.PCB_Ver,
                        checkerboard=opt.CB, share_conv=opt.share_conv)
            epochs = opt.epochs
        model.rpp = True
        model = model.convert_to_rpp()

        model = model.cuda()
        print(model)
        model = rpp_train(model, criterion, stage, epochs)

        # step4: whole net training #
        print("STARTING STEP 4 OF PCB:")
        stage = 'full'
        full_train(model, criterion, stage, 40)


else:
    # model to gpu
    model = model.cuda()
    print(model)
    if fp16:
        # model = network_to_half(model)
        # optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
        model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=opt.epochs)

