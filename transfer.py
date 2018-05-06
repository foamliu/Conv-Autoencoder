from vgg16 import vgg16_model
from utils import custom_loss
from utils import decoder
from utils import compile


def autoencoder(img_rows, img_cols, channel=3):
    model = vgg16_model(img_rows, img_cols, channel)
    model.layers.pop()  # dense_4
    model.layers.pop()  # dropout_2
    model.layers.pop()  # dense_2
    model.layers.pop()  # dropout_1
    model.layers.pop()  # dense_1
    model.layers.pop()  # flatten_1

    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    # Decoder
    model = decoder(model)

    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    print(model.summary())

    compile(model)
    return model


if __name__ == '__main__':
    model = autoencoder(224, 244, 3)
    model.save_weights('model_weights.h5')
