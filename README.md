# Fast Compressive Sensing Using Generative Model with Structed Latent Variables

## Introduction
This is a Tensorflow implementation of the paper ["Fast Compressive Sensing Using Generative Model with Structed Latent Variables"](http://arxiv.org/abs/1902.06913), available at http://arxiv.org/abs/1902.06913.

This code contains three major parts: our-pretrained neural networks (InfoGAN, DCGAN, DAE), the algorithm to reconstruct a signal from its compressed measurements, and a small train and test dataset cropped and resized from the CelebA dataset.
The Datasets/train.npy file is a small sample of training data. The complete training dataset can be found here. https://app.box.com/s/kj0dx1tio8yqcm0lg61nojkqvvljf3lz

## Prerequisites

We tested the code with python 2.7 and Tensorflow 1.12.0.

## Weights of Neural Networks
Please download the pre-trained neural networks from https://app.box.com/s/gwt4qdlhlaxve404jpt20hlmfe6sdky1 and place MNIST and CelebA in the folder Neural_Networks.

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

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/sihan-zeng/f-csrg/blob/master/LICENSE) file for details.


