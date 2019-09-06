#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-16 16:20:01

import sys
sys.path.append('./')
import numpy as np
import torch
from networks import VDN
from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_float, img_as_ubyte
from utils import load_state_dict_cpu
from matplotlib import pyplot as plt
import time
from scipy.io import loadmat

use_gpu = True
C = 3
dep_U = 4

# load the pretrained model
print('Loading the Model')
checkpoint = torch.load('./model_state/model_state_SIDD')
net = VDN.VDNU(C, dep_U=dep_U, wf=64, batch_norm=True)
if use_gpu:
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(checkpoint)
else:
    load_state_dict_cpu(net, checkpoint)
net.eval()

im_noisy = loadmat('./test_data/DND/1.mat')['im_noisy']
H, W, _ = im_noisy.shape
if H % 2**dep_U != 0:
    H -= H % 2**dep_U
if W % 2**dep_U != 0:
    W -= W % 2**dep_U
im_noisy = im_noisy[:H, :W, ]

im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])
if use_gpu:
    im_noisy = im_noisy.cuda()
    print('Begin Testing on GPU')
else:
    print('Begin Testing on CPU')
with torch.autograd.set_grad_enabled(False):
    tic = time.time()
    phi_Z = net(im_noisy, 'test')
    toc = time.time()
    err = phi_Z.cpu().numpy()
if use_gpu:
    im_noisy = im_noisy.cpu().numpy()
else:
    im_noisy = im_noisy.numpy()
im_denoise = im_noisy - err[:, :C,]
im_denoise = np.transpose(im_denoise.squeeze(), (1,2,0))
im_denoise = img_as_ubyte(im_denoise.clip(0,1))
im_noisy = np.transpose(im_noisy.squeeze(), (1,2,0))
im_noisy = img_as_ubyte(im_noisy.clip(0,1))

plt.subplot(121)
plt.imshow(im_noisy)
plt.title('Noisy Image')
plt.subplot(122)
plt.imshow(im_denoise)
plt.title('Denoised Image')
plt.show()
