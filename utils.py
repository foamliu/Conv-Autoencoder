import os

import cv2 as cv
import keras.backend as K
import numpy as np
from console_progressbar import ProgressBar
from keras.optimizers import SGD


def custom_loss(y_true, y_pred):
    epsilon = 1e-6
    epsilon_sqr = K.constant(epsilon ** 2)
    return K.mean(K.sqrt(K.square(y_pred - y_true) + epsilon_sqr))


def load_data():
    # (num_samples, 320, 320, 4)
    num_samples = 8144
    train_split = 0.8
    batch_size = 16
    num_train = int(round(num_samples * train_split))
    num_valid = num_samples - num_train
    pb = ProgressBar(total=100, prefix='Loading data', suffix='', decimals=3, length=50, fill='=')

    x_train = np.empty((num_train, 320, 320, 4), dtype=np.float32)
    y_train = np.empty((num_train, 320, 320, 1), dtype=np.float32)
    x_valid = np.empty((num_valid, 320, 320, 4), dtype=np.float32)
    y_valid = np.empty((num_valid, 320, 320, 1), dtype=np.float32)

    i_train = i_valid = 0
    for root, dirs, files in os.walk("data", topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            bgr_img = cv.imread(filename)
            gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            if filename.startswith('data/train'):
                x_train[i_train, :, :, 0:3] = rgb_img / 255.
                x_train[i_train, :, :, 3] = np.random.uniform(0, 1, (320, 320))
                y_train[i_train, :, :, 0] = gray_img / 255.
                i_train += 1
            elif filename.startswith('data/valid'):
                x_valid[i_valid, :, :, 0:3] = rgb_img / 255.
                x_valid[i_train, :, :, 3] = np.random.uniform(0, 1, (320, 320))
                y_valid[i_valid, :, :, 0] = gray_img / 255.
                i_valid += 1

            i = i_train + i_valid
            if i % batch_size == 0:
                pb.print_progress_bar(i * 100 / num_samples)

    return x_train, y_train, x_valid, y_valid


def do_compile(model):
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.99, nesterov=True)
    # model.compile(optimizer='nadam', loss=custom_loss)
    model.compile(optimizer=sgd, loss=custom_loss)
    return model
