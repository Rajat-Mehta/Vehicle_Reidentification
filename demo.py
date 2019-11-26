import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=0, type=int, help='test_image_index')
parser.add_argument('--use_siamese', action='store_true', help='use siamese')
parser.add_argument('--test_dir', default='../Datasets/VeRi_with_plate/pytorch',type=str, help='./test_data')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--use_ftnet', action='store_true', help='use siamese')
parser.add_argument('--veri_wild', action='store_true', help='use veri wild dataset')
parser.add_argument('--testing', action='store_true', help='use testing dataset')
parser.add_argument('--gpu_ids', default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--fusion', action='store_true', help='use fusion')
parser.add_argument('--model', default='pcb', type=str, help='test_image_index')

opts = parser.parse_args()
VERI_WILD = '../Datasets/VeRI-Wild/pytorch'
VERI = '../Datasets/VeRi_with_plate/pytorch'
TESTING = '../Datasets/Testing'
str_ids = opts.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
part = ''
if opts.use_ftnet:
    name = "ft_ResNet"
elif opts.use_siamese:
    name = "siamese"
elif opts.fusion:
    name = "fusion/finetuned_wild"
elif opts.PCB:
    if opts.model == 'ccb':
        name = "ft_ResNet_PCB/clustering/finetuned_cluster6"
    elif opts.model =='pcb_cb':
        name = "ft_ResNet_PCB/finetune_wild/wild_79_CB"
    elif opts.model =='pcb_finetuned':
        name = "ft_ResNet_PCB/finetune_wild/wild_79_vertical"
    else:
        name = "ft_ResNet_PCB/vertical/part6_vertical"

print("Loading feature vectors from this path: ", name)
    
data_dir = opts.test_dir
if opts.testing:
    data_dir = TESTING
    opts.test_dir = TESTING
    name = "ft_ResNet_PCB/features_testing"

if opts.veri_wild:
    data_dir = VERI_WILD
    opts.test_dir = VERI_WILD
    part = '_6001-10000'
    name = "ft_ResNet_PCB/features_veri_wild"

image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir, x) ) for x in ['gallery'+part,'query'+part]}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat(os.path.join('./model', name, 'pytorch_result_VeRi' + part +'.mat'))

query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

def get_index(gf, query, qf):
    if opts.use_siamese:
        qf = qf.unsqueeze_(0).repeat(len(gf), 1)
        distance = F.pairwise_distance(qf, gf, keepdim=True)
        distance = distance.squeeze(1).cpu()
        distance = distance.numpy()
        # predict index
        index = np.argsort(distance)  # from small to large
    else:
        
        score = torch.mm(gf, query)
        score = score.squeeze(1).cpu()
        score = score.numpy()
        
        # predict index
        index = np.argsort(score)  # from small to large
        index = index[::-1]
        # index = index[0:2000]
        # good index
    return index
#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1,1)
    # print(query.shape)
    index = get_index(gf, query, qf)
    query_index = np.argwhere(gl==ql)
    #same camera
    camera_index = np.argwhere(gc==qc)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)
    junk_index = np.append(junk_index, camera_index)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

i = opts.query_index
index = sort_img(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)

########################################################################
# Visualize the rank result

query_path, _ = image_datasets['query'+ part].imgs[i]
query_label = query_label[i]
print("Query label: ", query_label)
print('Top 10 images are as follow:')
try:  # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path,'query')
    for i in range(10):
        ax = plt.subplot(1, 11, i+2)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'+part].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d'%(i+1), color='green')
        else:
            ax.set_title('%d'%(i+1), color='red')
        print(img_path)
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("demo.png")
