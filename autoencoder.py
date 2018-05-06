from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from utils import custom_loss


def autoencoder(img_rows, img_cols, channel=1):
    model = Sequential()
    # Encoder
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
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

    # Decoder
    model.add(Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv6'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5_1'))
    model.add(Conv2D(512, (5, 5), activation='relu', padding='same', name='deconv5_2'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4_1'))
    model.add(Conv2D(256, (5, 5), activation='relu', padding='same', name='deconv4_2'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3_1'))
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same', name='deconv3_2'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2_1'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv2_2'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1_1'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', name='deconv1_2'))

    model.add(Conv2D(1, (5, 5), activation='sigmoid', padding='same', name='pred'))

    # Learning rate is changed to 0.001
    # sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adadelta', loss=custom_loss)
    return model
