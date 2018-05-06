import keras.backend as K
import numpy as np
import os
import cv2 as cv
from console_progressbar import ProgressBar


def custom_loss(y_true, y_pred):
    epsilon = 1e-6
    epsilon_sqr = K.constant(epsilon ** 2)
    return K.mean(K.sqrt(K.square(y_pred - y_true) + epsilon_sqr))


def load_data():
    # (num_samples, 224, 224, 3)
    num_samples = 8144
    train_split = 0.8
    batch_size = 16
    num_train = int(round(num_samples * train_split))
    num_valid = num_samples - num_train
    pb = ProgressBar(total=100, prefix='Loading data', suffix='', decimals=3, length=50, fill='=')

    x_train = np.empty((num_train, 224, 224, 3), dtype=np.float32)
    y_train = np.empty((num_train, 224, 224, 1), dtype=np.float32)
    x_valid = np.empty((num_valid, 224, 224, 3), dtype=np.float32)
    y_valid = np.empty((num_valid, 224, 224, 1), dtype=np.float32)

    i_train = i_valid = 0
    for root, dirs, files in os.walk("data", topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            bgr_img = cv.imread(filename)
            gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            if filename.startswith('data/train'):
                x_train[i_train, :, :, :] = rgb_img / 255.
                y_train[i_train, :, :, 0] = gray_img / 255.
                i_train += 1
            else:
                x_valid[i_valid, :, :, :] = rgb_img / 255.
                y_valid[i_valid, :, :, 0] = gray_img / 255.
                i_valid += 1

            i = i_train + i_valid
            if i % batch_size == 0:
                pb.print_progress_bar((i + 2) * 100 / num_samples)

    return x_train, y_train, x_valid, y_valid
