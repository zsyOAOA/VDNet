#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import torch
import h5py as h5
import random
import cv2
import os
import numpy as np
import torch.utils.data as uData
from skimage import img_as_float32 as img_as_float
from .data_tools import sigma_estimate, random_augmentation, gaussian_kernel
from . import BaseDataSetH5, BaseDataSetImg

# Benchmardk Datasets: Renoir and SIDD
class BenchmarkTrain(BaseDataSetH5):
    def __init__(self, h5_file, length, pch_size=128, radius=5, eps2=1e-6, noise_estimate=True):
        super(BenchmarkTrain, self).__init__(h5_file, length)
        self.win = 2*radius + 1
        self.sigma_spatial = radius
        self.noise_estimate = noise_estimate
        self.eps2 = eps2
        self.pch_size = pch_size

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[ind_im]]
            im_gt, im_noisy = self.crop_patch(imgs_sets)
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        if self.noise_estimate:
            sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))
        eps2 = torch.tensor([self.eps2], dtype=torch.float32).reshape((1,1,1))

        if self.noise_estimate:
            sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2, 0, 1)))
            return im_noisy, im_gt, sigma2_map_est, eps2
        else:
            return im_noisy, im_gt

class BenchmarkTest(BaseDataSetH5):
    def __getitem__(self, index):
        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2/2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt

# Simulation Datasets:
class SimulateTrain(BaseDataSetImg):
    def __init__(self, im_list, length,  pch_size=128, radius=5, noise_estimate=True):
        super(SimulateTrain, self).__init__(im_list, length,  pch_size)
        self.win = 2*radius + 1
        self.sigma_spatial = radius
        self.noise_estimate = noise_estimate
        self.num_images = len(im_list)
        self.sigma_min = 0
        self.sigma_max = 75

    def generate_sigma(self):
        pch_size = self.pch_size
        center = [random.uniform(0, pch_size), random.uniform(0, pch_size)]
        scale = random.uniform(pch_size/4, pch_size/4*3)
        kernel = gaussian_kernel(pch_size, pch_size, center, scale)
        up = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        down = random.uniform(self.sigma_min/255.0, self.sigma_max/255.0)
        if up < down:
            up, down = down, up
        up += 5/255.0
        sigma_map = down + (kernel-kernel.min())/(kernel.max()-kernel.min())  *(up-down)
        sigma_map = sigma_map.astype(np.float32)

        return sigma_map[:, :, np.newaxis]

    def __getitem__(self, index):
        pch_size = self.pch_size
        ind_im = random.randint(0, self.num_images-1)

        im_ori = cv2.imread(self.im_list[ind_im], 1)[:, :, ::-1]
        im_gt = img_as_float(self.crop_patch(im_ori))
        C = im_gt.shape[2]

        # generate sigmaMap
        sigma_map = self.generate_sigma()

        # generate noise
        noise = torch.randn(im_gt.shape).numpy() * sigma_map
        im_noisy = im_gt + noise.astype(np.float32)

        im_gt, im_noisy, sigma_map = random_augmentation(im_gt, im_noisy, sigma_map)

        if self.noise_estimate:
            sigma2_map_est = sigma_estimate(im_noisy, im_gt, self.win, self.sigma_spatial) #CxHxW
            sigma2_map_est = torch.from_numpy(sigma2_map_est.transpose((2,0,1)))
            # Groundtruth SigmaMap
            sigma2_map_gt = np.tile(np.square(sigma_map), (1, 1, C))
            sigma2_map_gt = np.where(sigma2_map_gt<1e-10, 1e-10, sigma2_map_gt)
            sigma2_map_gt = torch.from_numpy(sigma2_map_gt.transpose((2, 0, 1)))
        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        if self.noise_estimate:
            return im_noisy, im_gt, sigma2_map_est, sigma2_map_gt
        else:
            return im_noisy, im_gt

class SimulateTest(uData.Dataset):
    def __init__(self, im_list, h5_path):
        super(SimulateTest, self).__init__()
        self.im_list = im_list
        self.h5_path = h5_path

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        im_gt = cv2.imread(self.im_list[index], 1)[:, :, ::-1]
        im_key = os.path.basename(self.im_list[index]).split('.')[0]
        C = im_gt.shape[2]

        with h5.File(self.h5_path, 'r') as h5_file:
            noise = np.array(h5_file[im_key][:,:,:C])
        H, W, _ = noise.shape
        im_gt = img_as_float(im_gt[:H, :W])
        im_noisy = im_gt + noise

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1))).type(torch.float32)
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1))).type(torch.float32)

        return im_noisy, im_gt

