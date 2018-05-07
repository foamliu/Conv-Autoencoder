import keras.backend as K
from keras.layers import ZeroPadding2D, Conv2D
from keras.models import Sequential
import numpy as np
from utils import compile
from utils import decoder
from vgg16 import vgg16_model


def autoencoder(img_rows, img_cols, channel=4):
    old_model = vgg16_model(img_rows, img_cols, 3)
    old_model.layers.pop()  # dense_4
    old_model.layers.pop()  # dropout_2
    old_model.layers.pop()  # dense_2
    old_model.layers.pop()  # dropout_1
    old_model.layers.pop()  # dense_1
    old_model.layers.pop()  # flatten_1

    old_model.outputs = [old_model.layers[-1].output]
    old_model.layers[-1].outbound_nodes = []

    # Decoder
    old_model = decoder(old_model)

    old_model.outputs = [old_model.layers[-1].output]
    old_model.layers[-1].outbound_nodes = []

    print(old_model.summary())

    # Add a new channel for trimap
    old_conv1_1 = old_model.get_layer('conv1_1')
    old_weights = old_conv1_1.get_weights()[0]
    old_biases = old_conv1_1.get_weights()[1]
    old_layers = [l for l in old_model.layers]

    new_input_layer = ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel), name='input')
    new_weights = np.zeros((3, 3, channel, 64), dtype=np.float32)
    new_biases = old_biases
    new_weights[:, :, 0:3, :] = old_weights
    new_weights[:, :, 3:channel, :] = 0.0
    new_conv1_1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1')

    new_model = Sequential()
    new_model.add(new_input_layer)
    new_model.add(new_conv1_1)
    new_conv1_1.set_weights([new_weights, new_biases])


    for i in range(2, len(old_layers)):
        new_model.add(old_layers[i])

    # new_model.outputs = [new_model.layers[-1].output]
    # new_model.layers[-1].outbound_nodes = []

    print(new_model.summary())

    compile(new_model)
    return old_model


if __name__ == '__main__':
    model = autoencoder(224, 224, 4)
    input_layer = model.get_layer('input')
    print(input_layer.input_shape)
    print(input_layer.output_shape)
    conv1_1 = model.get_layer('conv1_1')
    weights = conv1_1.get_weights()
    print(len(weights))
    print(weights[0].shape)
    print(weights[1].shape)
    print(conv1_1.input)
    print(conv1_1.output)
    print(conv1_1.input_shape)
    print(conv1_1.output_shape)
    # model.save_weights('model_weights.h5')

    K.clear_session()
