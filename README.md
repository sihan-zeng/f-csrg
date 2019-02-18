# Fast Compressive Sensing Using Generative Model with Structed Latent Variables

## Introduction
This is a Tensorflow implementation of the paper "Fast Compressive Sensing Using Generative Model with Structed Latent Variables".

This code contains three major parts: our-pretrained neural networks (InfoGAN, DCGAN, DAE), the algorithm to reconstruct a signal from its compressed measurements, and a small train and test dataset cropped and resized from the CelebA dataset.
The Datasets/train.npy file is a small sample of training data. The complete training dataset can be found here. https://gatech.box.com/s/v0zjofvvpv3vuoyd5mmsfqpiyea2sd4u

## Prerequisites

We tested the code with python 2.7 and Tensorflow 1.12.0.

## Weights of Neural Networks
Please download the pre-trained neural networks from https://gatech.box.com/s/gdteqz2te9x38yuurj70i4joiowanteo and place MNIST and CelebA in the folder Neural_Networks.

## Run Compressive Sensing Recovery

To test on MNIST dataset,

```
cd Compressed_Domain_Processing/MNIST/Recovery

python method_comparison.py 
```


To test on CelebA dataset,

```
cd Compressed_Domain_Processing/CelebA/Recovery

python method_comparison.py 
```


## License

This project is licensed under the MIT License - see the LICENSE.md file for details.


