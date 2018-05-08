import os

import cv2 as cv
import keras.backend as K
import numpy as np
from console_progressbar import ProgressBar
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D


def custom_loss(y_true, y_pred):
    epsilon = 1e-6
    epsilon_sqr = K.constant(epsilon ** 2)
    return K.mean(K.sqrt(K.square(y_pred - y_true) + epsilon_sqr))


def load_data():
    # (num_samples, 224, 224, 4)
    num_samples = 8144
    train_split = 0.8
    batch_size = 16
    num_train = int(round(num_samples * train_split))
    num_valid = num_samples - num_train
    pb = ProgressBar(total=100, prefix='Loading data', suffix='', decimals=3, length=50, fill='=')

    x_train = np.empty((num_train, 224, 224, 4), dtype=np.float32)
    y_train = np.empty((num_train, 224, 224, 1), dtype=np.float32)
    x_valid = np.empty((num_valid, 224, 224, 4), dtype=np.float32)
    y_valid = np.empty((num_valid, 224, 224, 1), dtype=np.float32)

    i_train = i_valid = 0
    for root, dirs, files in os.walk("data", topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            bgr_img = cv.imread(filename)
            gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            if filename.startswith('data/train'):
                x_train[i_train, :, :, 0:3] = rgb_img / 255.
                x_train[i_train, :, :, 3] = np.random.uniform(0, 1, (224, 224))
                y_train[i_train, :, :, 0] = gray_img / 255.
                i_train += 1
            else:
                x_valid[i_valid, :, :, 0:3] = rgb_img / 255.
                x_valid[i_train, :, :, 3] = np.random.uniform(0, 1, (224, 224))
                y_valid[i_valid, :, :, 0] = gray_img / 255.
                i_valid += 1

            i = i_train + i_valid
            if i % batch_size == 0:
                pb.print_progress_bar(i * 100 / num_samples)

    return x_train, y_train, x_valid, y_valid


def build_decoder(model):
    model.add(Conv2D(4096, (7, 7), activation='relu', padding='valid', name='conv6'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(7, 7)))

    model.add(
        Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6', kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5', kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))

    model.add(
        Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(BatchNormalization())

    model.add(Conv2D(1, (5, 5), activation='sigmoid', padding='same', name='pred', kernel_initializer='he_normal',
                     bias_initializer='zeros'))


def build_encoder(model, img_rows, img_cols, channel):
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel), name='input'))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


# def encoder_bn(model, img_rows, img_cols, channel):
#     model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
#     model.add(Conv2D(64, (3, 3), name='conv1_1'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(64, (3, 3), name='conv1_2'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(128, (3, 3), name='conv2_1'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(128, (3, 3), name='conv2_2'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(256, (3, 3), name='conv3_1'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(256, (3, 3), name='conv3_2'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(256, (3, 3), name='conv3_3'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(512, (3, 3), name='conv4_1'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(512, (3, 3), name='conv4_2'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(512, (3, 3), name='conv4_3'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(512, (3, 3), name='conv5_1'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(512, (3, 3), name='conv5_2'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(ZeroPadding2D((1, 1)))
#     model.add(Conv2D(512, (3, 3), name='conv5_3'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#     return model


def compile(model):
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.99, nesterov=True)
    model.compile(optimizer='nadam', loss=custom_loss)
    return model
