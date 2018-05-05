# Convolutional Autoencoder

This repository is to do convolutional autoencoder by fine-tuning SetNet with Cars Dataset from Stanford.


## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split.

 ![image](https://github.com/foamliu/Conv-Autoencoder/raw/master/images/random.jpg)

You can get it from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

```bash
$ cd Car-Recognition
$ wget http://imagenet.stanford.edu/internal/car196/cars_train.tgz
$ wget http://imagenet.stanford.edu/internal/car196/cars_test.tgz
$ wget --no-check-certificate https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
```

## Architecture

![image](https://github.com/foamliu/Conv-Autoencoder/raw/master/images/segnet.jpg)


## ImageNet Pretrained Models

Download [VGG16](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) into models folder.


## Usage

### Data Pre-processing
Extract 8,144 training images, and split them by 80:20 rule (6,515 for training, 1,629 for validation):
```bash
$ python pre-process.py
```

### Train
```bash
$ python train.py
```

### Demo
Download [pre-trained model] into "models" folder then run:

```bash
$ python demo.py --i [image_path]
```

#### Sample 1

![image](https://github.com/foamliu/Conv-Autoencoder/raw/master/images/sample-1.jpg)
 
#### Sample 2

![image](https://github.com/foamliu/Conv-Autoencoder/raw/master/images/sample-2.jpg)
 
#### Sample 3

![image](https://github.com/foamliu/Conv-Autoencoder/raw/master/images/sample-3.jpg)