import keras.backend as K
from keras.models import Sequential

from utils import compile
from utils import decoder, encoder


def autoencoder(img_rows, img_cols, channel=4):
    model = Sequential()
    # Encoder
    model = encoder(model, img_rows, img_cols, channel)
    # Decoder
    model = decoder(model)
    compile(model)
    return model


if __name__ == '__main__':
    model = autoencoder(224, 224, 4)
    input_layer = model.get_layer('input')

    K.clear_session()
