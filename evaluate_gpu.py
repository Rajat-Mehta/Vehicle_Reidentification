import scipy.io
import torch
import numpy as np
#import time
import os
import random
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Evaluate')
parser.add_argument('--use_siamese',  action='store_true', help='evaluate siamese or not')
parser.add_argument('--keep_num',  default=100, help='how many images to keep from other cameras')
parser.add_argument('--use_single_camera',  action='store_true', help='use single camera gallery images')
parser.add_argument('--PCB',  action='store_true', help='evaluate PCB or not')
parser.add_argument('--siamese_PCB',  action='store_true', help='evaluate Siamese with PCB or not')

opts = parser.parse_args()
KEEP_NUM = int(opts.keep_num)
gallery_size = 0
# Evaluate

if opts.use_siamese:
    name = 'siamese'
elif opts.siamese_PCB:
    name = 'siamese_PCB'
elif opts.PCB:
    name = 'ft_ResNet_PCB'
else:
    name = 'ft_ResNet'


def evaluate_single_camera(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    same_camera_index = np.argwhere(gc == qc)
    same_camera_index = same_camera_index.squeeze()

    gl = gl[same_camera_index]
    gc = gc[same_camera_index]
    gf = gf[same_camera_index]

    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index

    good_index = np.argwhere(gl == ql)

    junk_index1 = np.argwhere(gl==-1)
    good_index = np.setdiff1d(good_index, junk_index1, assume_unique=True)

    CMC_tmp = compute_mAP(index, good_index, junk_index1)
    return CMC_tmp


def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf.view(-1,1)
    # print(query.shape)

    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)

    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)

    junk_index = np.append(junk_index2, junk_index1) #.flatten())

    # limit the number of images per camera/view in gallery set to (max) keep_num
    new_junk = limit_gallery_images(ql, qc,  gl, gc, junk_index, KEEP_NUM)
    junk_index = np.append(junk_index, new_junk)

    # remove all images from same camera as query_camera
    junk_index = np.append(junk_index, camera_index)
    good_index = np.setdiff1d(good_index, junk_index, assume_unique=True)
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def find_distance(feat1, feat2):
    return F.pairwise_distance(feat1, feat2, keepdim=True)


def evaluate_siamese_single_camera(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    same_camera_index = np.argwhere(gc == qc)
    same_camera_index = same_camera_index.squeeze()

    gl = gl[same_camera_index]
    gc = gc[same_camera_index]
    gf = gf[same_camera_index]

    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    qf = qf.unsqueeze_(0).repeat(len(gf), 1)
    distance = F.pairwise_distance(gf, qf, keepdim=True)

    distance = distance.squeeze(1).cpu()
    distance = distance.numpy()

    index = np.argsort(distance)  # from small to large

    # good index

    good_index = np.argwhere(gl == ql)

    junk_index1 = np.argwhere(gl==-1)
    good_index = np.setdiff1d(good_index, junk_index1, assume_unique=True)

    CMC_tmp = compute_mAP(index, good_index, junk_index1)
    return CMC_tmp


def evaluate_siamese(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)

    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()

    qf = qf.unsqueeze_(0).repeat(len(gf), 1)
    distance = F.pairwise_distance(gf, qf, keepdim=True)

    distance = distance.squeeze(1).cpu()
    distance = distance.numpy()

    # predict index
    index = np.argsort(distance)  # from small to large
    # index = index[::-1]

    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)

    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    # limit the number of images per camera/view in gallery set to (max) keep_num
    new_junk = limit_gallery_images(ql, qc, gl, gc, junk_index, KEEP_NUM)
    junk_index = np.append(junk_index, new_junk)

    # remove all images from same camera as query_camera
    junk_index = np.append(junk_index, camera_index)
    good_index = np.setdiff1d(good_index, junk_index, assume_unique=True)

    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def limit_gallery_images(ql, qc, gl, gc, junk_index, keep_num):
    """ limit the number of images per camera/view in gallery set to (max) keep_num """

    good_gallery_camera = np.intersect1d(np.argwhere(gl == ql), np.argwhere(gc != qc))
    good_gallery_camera = np.setdiff1d(good_gallery_camera, junk_index)

    camera_labels = gc[good_gallery_camera]

    to_remove = []
    for j in set(gc[good_gallery_camera]):

        if list(camera_labels).count(j) > keep_num:
            temp = np.where(camera_labels == j)[0]
            rand = np.random.choice(temp, keep_num, replace=False)
            diff = list(np.setdiff1d(temp, rand))
            to_remove.extend(diff)

    new_good_gallery_camera = np.delete(good_gallery_camera, to_remove)
    junk = np.setdiff1d(good_gallery_camera, new_good_gallery_camera)

    return junk


def compute_mAP(index, good_index, junk_index):
    # good_index can be seen as ground truth images for given query image

    ap = 0
    cmc = torch.IntTensor(gallery_size).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    # rows_good can be considered as correct predictions out of all of the predictions
    # CONFUSION
    cmc[rows_good[0]:] = 1

    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2
    return ap, cmc


######################################################################
feat_path = './model/' + name + '/pytorch_result_VeRi.mat'
result = scipy.io.loadmat(feat_path)
print("Loading saved features from:", feat_path)
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('./model/' + name + '/multi_query.mat')


if multi:
    m_result = scipy.io.loadmat('./model/' + name + '/multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()


print(query_feature.shape)
print(gallery_feature.shape)

CMC = torch.IntTensor(len(gallery_label)).zero_()
gallery_size = len(gallery_label)
ap = 0.0
for i in range(len(query_label)):
    if opts.use_single_camera and opts.use_siamese:
        ap_tmp, CMC_tmp = evaluate_siamese_single_camera(query_feature[i], query_label[i], query_cam[i],
                                                         gallery_feature, gallery_label, gallery_cam)

    elif opts.use_siamese:
        ap_tmp, CMC_tmp = evaluate_siamese(query_feature[i], query_label[i], query_cam[i],
                                           gallery_feature, gallery_label, gallery_cam)
    elif opts.use_single_camera:
        ap_tmp, CMC_tmp = evaluate_single_camera(query_feature[i], query_label[i], query_cam[i],
                                   gallery_feature, gallery_label, gallery_cam)


    else:
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i], query_cam[i],
                                   gallery_feature, gallery_label, gallery_cam)
    if CMC_tmp[0]==-1:
        continue

    CMC = CMC + CMC_tmp
    ap += ap_tmp
    #print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))

result = open("./results/result_other_cams" + str(KEEP_NUM) + ".txt", "w")
result.write('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
result.close()

# multiple-query
CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0
if multi:
    for i in range(len(query_label)):
        mquery_index1 = np.argwhere(mquery_label==query_label[i])
        mquery_index2 = np.argwhere(mquery_cam==query_cam[i])
        mquery_index =  np.intersect1d(mquery_index1, mquery_index2)
        mq = torch.mean(mquery_feature[mquery_index,:], dim=0)
        ap_tmp, CMC_tmp = evaluate(mq,query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        #print(i, CMC_tmp[0])
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC

    print('multi Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
