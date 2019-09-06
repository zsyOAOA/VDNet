#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-04-17 22:43:24

import os

_base_path = 'G:/Dataset/SIDD/Full/'
urls_path = _base_path + 'SIDD_URLs.txt'

# camera type: S6, GP, IP, N6, G4
camera_type = ['S6', 'GP', 'G4']
for camera in camera_type:
    camera_dir = _base_path + camera + '/'
    if not os.path.isdir(camera_dir):
        os.mkdir(camera_dir)
    with open(urls_path, 'r') as f_url:
        num_url = 0
        for url in f_url:
            if (camera in url) and ('SRGB' in url):
                num_url += 1
                print('\n{:d}: {:s}'.format(num_url, url.split('/')[-2]))
                cmd = 'axel -n 32 -a -o ' + camera_dir + url.split('/')[-1][:-1] + ' ' + url[:-1]
                os.system(cmd)


