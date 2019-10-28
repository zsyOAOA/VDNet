#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-05 14:44:04

import sys
sys.path.append('./')

import numpy as np
import cv2
from skimage import img_as_float
from utils import generate_gauss_kernel_mix, peaks, sincos_kernel
import h5py as h5
from pathlib import Path
import argparse

base_path = Path('test_data')
seed = 10000

np.random.seed(seed)
kernels = [peaks(256), sincos_kernel(), generate_gauss_kernel_mix(256, 256)]
dep_U = 4

sigma_max = 75/255.0
sigma_min = 10/255.0
for data_name in ['LIVE1', 'Set5', 'CBSD68']:
    if data_name == 'LIVE1' or data_name == 'Set5':
        im_list = sorted((base_path / data_name).glob('*.bmp'))
    else:
        im_list = sorted((base_path / data_name).glob('*.png'))

    im_list = sorted([x.name for x in im_list])

    for jj, sigma in enumerate(kernels):
        print('Case {:d} of Dataset {:s}: {:d} images'.format(jj+1, data_name, len(im_list)))
        # generate sigmaMap
        sigma = sigma_min + (sigma-sigma.min())/(sigma.max()-sigma.min()) * (sigma_max-sigma_min)
        noise_dir = base_path / 'noise_niid'
        if not noise_dir.is_dir():
            noise_dir.mkdir()
        h5_path = noise_dir.joinpath(data_name + '_niid_case' + str(jj+1) + '.hdf5')
        if h5_path.exists():
            h5_path.unlink()
        with h5.File(h5_path) as h5_file:
            for ii, im_name in enumerate(im_list):
                gt_name = str(base_path / data_name / im_name)
                im_gt = cv2.imread(gt_name, 1)[:, :, ::-1]
                H, W, C = im_gt.shape
                H -= int(H % pow(2, dep_U))
                W -= int(W % pow(2, dep_U))
                im_gt = img_as_float(im_gt[:H, :W, ])

                sigma = cv2.resize(sigma, (W, H))
                sigma = sigma.astype(np.float32)
                noise = np.random.randn(H, W, C) * np.expand_dims(sigma, 2)
                noise = noise.astype(np.float32)
                data = np.concatenate((noise, sigma[:,:,np.newaxis]), axis=2)
                h5_file.create_dataset(name=im_name.split('.')[0], dtype=data.dtype,
                                                                        shape=data.shape, data=data)

