from keras.models import Sequential
from utils import decoder, encoder
from utils import compile


def autoencoder(img_rows, img_cols, channel=1):
    model = Sequential()
    # Encoder
    model = encoder(model, img_rows, img_cols, channel)
    # Decoder
    model = decoder(model)
    compile(model)
    return model
