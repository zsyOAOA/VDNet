# Variational Denoising Network: Toward Blind Noise Modeling and Removal (NeurIPS, 2019) [arXiv](https://arxiv.org/pdf/1908.11314v2.pdf)
# Requirements and Dependencies
* Ubuntu 16.04, cuda 10.0
* Python 3.6, Pytorch 1.1.0
* More detail (See environment.yml)

# Image Denoising
## Non-IID Gauss Noise Removal
* In the training stage, we used the source images in [Waterloo](https://ece.uwaterloo.ca/~k29ma/exploration/),
[CBSD432 and CImageNet400](https://drive.google.com/folderview?id=0B-_yeZDtQSnobXIzeHV5SjY5NzA&usp=sharing) as groundtruth,
and the variance map of the noise was generated with Gaussian kernel. However, in order to test the generalization of our model,
three different variance maps were adopted to generate the noisy images of the three groups of testing datasets. The four kinds
of variance maps are shown in the following:

<img src="./figs/sigmaMap.png" align=center />

* Testing our trained model, please run the demo:
```
    python demo_test_simulation.py
```
  and get the following denoising results:

<img src="./figs/simulation.eps" align=center />

* If you want to re-train our model, please follow these three steps:

    1. Download the source images from the above links.
    2. Prepare the testing datasets:
    ```
        python datasets/prepare_data/simulation/noise_generate_nips_niid.py
    ```
    3. Training:
    ```
        python train_simulation.py --simulate_dir source_imgs_path --eps2 5e-5
    ```

## Real-world Noise Removal

The real-world denoiser was trained using the [SIDD Medium Dataset](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php). For re-training VDN, follow these steps:
1. Download the [training](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Medium_Srgb.zip) and validation([noisy](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationNoisyBlocksSrgb.mat), [groundtruth](ftp://sidd_user:sidd_2018@130.63.97.225/SIDD_Blocks/ValidationGtBlocksSrgb.mat)) datasets, and put the unzipped training
dataset and the validation dataset into the floder "sidd_data_path".

2. Writing the training and validation datasets into hdf5 fromat:
```
    python datasets/prepare_data/SIDD/big2small_train.py --data_dir sidd_data_path
    python datasets/prepare_data/SIDD/big2small_test.py --data_dir sidd_data_path
```
3. Training:
    python train_simulation.py --SIDD_dir sidd_data_path --eps2 1e-6

