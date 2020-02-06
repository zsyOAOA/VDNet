#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-16 16:20:01

import sys
sys.path.append('./')
import cv2
import numpy as np
import torch
from networks import VDN
from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_float, img_as_ubyte
from pathlib import Path
from utils import peaks, sincos_kernel, generate_gauss_kernel_mix, load_state_dict_cpu
from matplotlib import pyplot as plt
import time

use_gpu = False
case = 2
C = 3
dep_U = 4

# load the pretrained model
print('Loading the Model')
checkpoint = torch.load('./model_state/model_state_niidgauss')
net = VDN(C, dep_U=dep_U, wf=64)
if use_gpu:
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(checkpoint)
else:
    load_state_dict_cpu(net, checkpoint)
net.eval()

# im_path = str(Path('test_data') / 'CBSD68' / '101087.png')
im_path = str(Path('test_data') / 'CBSD68' / '285079.png')
im_name = im_path.split('/')[-1]
im_gt = img_as_float(cv2.imread(im_path)[:, :, ::-1])
H, W, _ = im_gt.shape
if H % 2**dep_U != 0:
    H -= H % 2**dep_U
if W % 2**dep_U != 0:
    W -= W % 2**dep_U
im_gt = im_gt[:H, :W, ]

# Generate the sigma map
if case == 1:
    # Test case 1
    sigma = peaks(256)
elif case == 2:
    # Test case 2
    sigma = sincos_kernel()
elif case == 3:
    # Test case 3
    sigma = generate_gauss_kernel_mix(256, 256)
else:
    sys.exit('Please input the corrected test case: 1, 2 or 3')

sigma = 10/255.0 + (sigma-sigma.min())/(sigma.max()-sigma.min()) * ((75-10)/255.0)
sigma = cv2.resize(sigma, (W, H))
noise = np.random.randn(H, W, C) * sigma[:, :, np.newaxis]
im_noisy = (im_gt + noise).astype(np.float32)

im_noisy = torch.from_numpy(im_noisy.transpose((2,0,1))[np.newaxis,])
if use_gpu:
    im_noisy = im_noisy.cuda()
    print('Begin Testing on GPU')
else:
    print('Begin Testing on CPU')
with torch.autograd.set_grad_enabled(False):
    torch.cuda.synchronize()
    tic = time.perf_counter()
    phi_Z = net(im_noisy, 'test')
    torch.cuda.synchronize()
    toc = time.perf_counter()
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
im_gt = img_as_ubyte(im_gt)
psnr_val = compare_psnr(im_gt, im_denoise, data_range=255)
ssim_val = compare_ssim(im_gt, im_denoise, data_range=255, multichannel=True)

print('Image name: {:s}, PSNR={:5.2f}, SSIM={:7.4f}, time={:.4f}'.format(im_name, psnr_val,
                                                                                 ssim_val, toc-tic))

plt.subplot(131)
plt.imshow(im_gt)
plt.title('Groundtruth')
plt.subplot(132)
plt.imshow(im_noisy)
plt.title('Noisy Image')
plt.subplot(133)
plt.imshow(im_denoise)
plt.title('Denoised Image')
plt.show()

