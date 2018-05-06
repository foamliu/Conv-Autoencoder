# -*- coding: utf-8 -*-

from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from utils import encoder


def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    model = Sequential()
    model = encoder(model, img_rows, img_cols, channel)

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    weights_path = 'models/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path)

    return model
