#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-05 11:44:29

import argparse

def set_opts():
    parser = argparse.ArgumentParser()
    # trainning settings
    parser.add_argument('--batch_size', type=int, default=64,
                                                         help="Batchsize of training, (default:64)")
    parser.add_argument('--patch_size', type=int, default=128,
                                                   help="Patch size of data sample,  (default:128)")
    parser.add_argument('--epochs', type=int, default=60, help="Training epohcs,  (default:60)")
    parser.add_argument('--lr', type=float, default=2e-4,
                                                  help="Initialized learning rate, (default: 2e-4)")
    parser.add_argument('--gamma', type=float, default=0.5,
                                         help="Decaying rate for the learning rate, (default: 0.5)")
    parser.add_argument('-p', '--print_freq', type=int, default=100,
                                                              help="Print frequence (default: 100)")
    parser.add_argument('-s', '--save_model_freq', type=int, default=1,
                                                            help="Save moel frequence (default: 1)")

    # Cliping the Gradients Norm during the training
    parser.add_argument('--clip_grad_D', type=float, default=1e4,
                                             help="Cliping the gradients for D-Net, (default: 1e4)")
    parser.add_argument('--clip_grad_S', type=float, default=1e3,
                                             help="Cliping the gradients for S-Net, (default: 1e3)")

    # GPU settings
    parser.add_argument('--gpu_id', type=int, nargs='+', default=0,
                                                           help="GPU ID, which allow multiple GPUs")

    # dataset settings
    parser.add_argument('--SIDD_dir', default='/ssd1t/SIDD/', type=str, metavar='PATH',
                                              help="Path to save the SIDD dataset, (default: None)")
    parser.add_argument('--simulate_dir', default='/ssd1t/simulation/train', type=str,
                                    metavar='PATH', help="Path to save the images, (default: None)")

    # model and log saving
    parser.add_argument('--log_dir', default='./log', type=str, metavar='PATH',
                                                 help="Path to save the log file, (default: ./log)")
    parser.add_argument('--model_dir', default='./model', type=str, metavar='PATH',
                                             help="Path to save the model file, (default: ./model)")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                               help="Path to the latest checkpoint (default: None)")
    parser.add_argument('--num_workers', default=8, type=int,
                                                help="Number of workers to load data, (default: 8)")
    # hyper-parameters
    parser.add_argument('--eps2', default=1e-6, type=float,
                                                    help="Variance for prior of Z, (default: 1e-6)")
    parser.add_argument('--radius', default=3, type=int,
                                                help="Radius for the Gaussian filter, (default: 3)")

    # network architecture
    parser.add_argument('--net', type=str, default='VDN',
                               help="Network architecture: VDN, VDNRD or VDNRDU, (default:VDN)")
    parser.add_argument('--slope', type=float, default=0.2,
                                                 help="Initial value for LeakyReLU, (default: 0.2)")
    parser.add_argument('--wf', type=int, default=64,
                                                   help="Initilized filters of UNet, (default: 64)")
    parser.add_argument('--depth', type=int, default=4, help="The depth of UNet, (default: 4)")
    args = parser.parse_args()

    return args
