import keras
import keras.backend as K
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.datasets import mnist
from keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D
from keras.models import Model


def custom_loss(y_true, y_pred):
    epsilon = 1e-6
    epsilon_sqr = K.constant(epsilon ** 2)
    return K.mean(K.sqrt(K.square(y_pred - y_true) + epsilon_sqr))


if __name__ == '__main__':
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    autoencoder.compile(optimizer='nadam', loss=custom_loss)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.reshape(x_train, (-1, 28, 28, 1))
    x_train = x_train / 255.

    x_test = np.reshape(x_test, (-1, 28, 28, 1))
    x_test = x_test / 255.

    print(autoencoder.summary())

    # Callbacks
    patience = 50
    epochs = 10000

    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_loss:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    autoencoder.fit(x_train, x_train,
                    validation_data=(x_test, x_test),
                    epochs=epochs,
                    batch_size=128,
                    shuffle=True,
                    verbose=1,
                    callbacks=callbacks)

    K.clear_session()
