# VDNet
# Variational Denoising Network: Toward Blind Noise Modeling and Removal (NeurIPS, 2019) [arXiv](https://arxiv.org/pdf/1908.11314v2.pdf)
# Requirements and Dependencies
* Ubuntu 16.04, cuda 10.0
* Python 3.6, Pytorch 1.1.0
* More detail (See environment.yml)

# Testing
1. For the Non-IID Gaussian Denosing as described in the paper, please run [demo_test_simulation.py](demo_test_simulation.py). 
2. For real-wormd image denoising task, please run [demo_test_benckmark.py](demo_test_benchmark.py). The model was trained on the SIDD Medium Dataset (320 noisy and clean paris).

# Training
1. Prepare the dataset following the code in the floder [datasets](datasets). Data link: [Waterloo Exploration Database](https://ece.uwaterloo.ca/~k29ma/exploration/), [CBSD432 and CImageNet400](https://drive.google.com/folderview?id=0B-_yeZDtQSnobXIzeHV5SjY5NzA&usp=sharing).
2. Train VDN using [train_simulation.py](train_simulation.py) or [train_benchmark.py](train_benchmark.py).






