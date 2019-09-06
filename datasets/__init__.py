#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:35:24

import random
import cv2
import numpy as np
import torch.utils.data as uData
import h5py as h5

# Base Datasets
class BaseDataSetImg(uData.Dataset):
    def __init__(self, im_list, length, pch_size=128):
        '''
        Args:
            im_list (list): path of each image
            length (int): length of Datasets
            pch_size (int): patch size of the cropped patch from each image
        '''
        super(BaseDataSetImg, self).__init__()
        self.im_list = im_list
        self.length = length
        self.pch_size = pch_size
        self.num_images = len(im_list)

    def __len__(self):
        return self.length

    def crop_patch(self, im):
        H = im.shape[0]
        W = im.shape[1]
        if H < self.pch_size or W < self.pch_size:
            H = max(self.pch_size, H)
            W = max(self.pch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
        pch = im[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size]
        return pch

class BaseDataSetH5(uData.Dataset):
    def __init__(self, h5_path, length=None):
        '''
        Args:
            h5_path (str): path of the hdf5 file
            length (int): length of Datasets
        '''
        super(BaseDataSetH5, self).__init__()
        self.h5_path = h5_path
        self.length = length
        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)

    def __len__(self):
        if self.length == None:
            return self.num_images
        else:
            return self.length

    def crop_patch(self, imgs_sets):
        H, W, C2 = imgs_sets.shape
        C = int(C2/2)
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
        im_noisy = np.array(imgs_sets[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size, :C])
        im_gt = np.array(imgs_sets[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size, C:])
        return im_gt, im_noisy
