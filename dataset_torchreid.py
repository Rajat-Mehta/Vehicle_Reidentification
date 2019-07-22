from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import torchreid
import os
import os.path as osp

from torchreid.data import ImageDataset

DATASET_QUERY = '../Datasets/VeRi_with_plate/image_query/'
DATASET_TEST = '../Datasets/VeRi_with_plate/image_test/'
DATASET_TRAIN = '../Datasets/VeRi_with_plate/image_train/'


class VeRiDataset(ImageDataset):
    dataset_dir = 'veri_dataset'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        train = self.get_lst_data(DATASET_TRAIN, "train")
        gallery = self.get_lst_data(DATASET_TEST, "test")
        query = self.get_lst_data(DATASET_QUERY, "query")

        super(VeRiDataset, self).__init__(train, query, gallery, **kwargs)

    def get_lst_data(self, path, data_type):
        data = os.listdir(path)
        lst = []
        for item in data:
            img_path = path + item
            pid = int(item.split('_')[0])
            camid = int(item.split('_')[1][1:])
            tp = (img_path, pid, camid)
            lst.append(tp)
        lst = self.bring_lables_in_range(lst, data_type)
        return lst

    def bring_lables_in_range(self, lst, data_type):
        if data_type == "train":
            self.label_dict_train = self.get_label_dict(lst)
            label_dict = self.label_dict_train
        elif data_type == "test":
            self.label_dict_test = self.get_label_dict(lst)
            label_dict = self.label_dict_test
        elif data_type == "query":
            self.label_dict_query = self.label_dict_test
            label_dict = self.label_dict_query

        for i, item in enumerate(lst):
            tpl_lst = list(item)
            tpl_lst[1] = label_dict[tpl_lst[1]]
            item = tuple(tpl_lst)
            lst[i] = item

        return lst

    def get_label_dict(self, lst):
        labelset = []
        for item in lst:
            labelset.append(item[1])
        labelset = set(labelset)
        label_dict = {label: index for index, label in enumerate(labelset)}

        return label_dict
