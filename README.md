# VDN
# Variational Denoising Network: Toward Blind Noise Modeling and Removal (NeurIPS, 2019) [arXiv](https://arxiv.org/pdf/1908.11314v2.pdf)
# Requirements and Dependencies
* Ubuntu 16.04, cuda 10.0
* Python 3.6, Pytorch 1.1.0
* More detail (See environment.yml)

# Image Denoising
## Non-IID Gauss Noise Removal

In the training stage, we used the source images in [Waterloo](https://ece.uwaterloo.ca/~k29ma/exploration/),
[CBSD432 and CImageNet400](https://drive.google.com/folderview?id=0B-_yeZDtQSnobXIzeHV5SjY5NzA&usp=sharing) as groundtruth,
and the variance map of the noise was generated with Gaussian kernel. However, in the test stage,
three different variance maps were adopted to generate noisy images to verify the generalization of VDN.

<img src="./figs/sigmaMap.png" align=center />

### Testing
```
    python demo_test_simulation.py
```

<img src="./figs/simulation.png" align=center />

### Training

1. Download the source images from the above links.
2. Prepare the testing datasets:
```
    python datasets/prepare_data/simulation/noise_generate_nips_niid.py
```
3. Begining training:
```
    python train_simulation.py --simulate_dir source_imgs_path --eps2 5e-5
```

## Real-world Noise Removal

The real-world VDN denoiser was trained on the [SIDD Medium Dataset](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php), and directly tested on the SIDD and DND benchmarks.
### Testing:
```
    python demo_test_benchmark.py
```

<img src="./figs/DND_denoise.png" align=center />

### Training:
1. Download the [training](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Medium_Srgb.zip) and validation([noisy](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationNoisyBlocksSrgb.mat), [groundtruth](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationGtBlocksSrgb.mat)) datasets, and put the unzipped training
dataset and the validation dataset into the floder "sidd_data_path".

2. Writing the training and validation datasets into hdf5 fromat:
```
    python datasets/prepare_data/SIDD/big2small_train.py --data_dir sidd_data_path
    python datasets/prepare_data/SIDD/big2small_test.py --data_dir sidd_data_path
```
3. Begin training:
```
    python train_simulation.py --SIDD_dir sidd_data_path --eps2 1e-6
```
