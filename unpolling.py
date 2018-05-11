import keras
import keras.backend as K
from keras.layers import Input, Conv2D, UpSampling2D, Reshape, Permute, Activation, MaxPooling2D
from keras.models import Model

if __name__ == '__main__':
    h_image = w_image = 28
    channel = 1
    n_classes = 10
    img_input = Input(shape=(h_image, w_image, channel))
    x = Conv2D(512, (3, 3), padding='same')(img_input)
    orig = x  # Save output x
    x = MaxPooling2D()(x)

    x = UpSampling2D()(x)

    model = Model(inputs=img_input, outputs=x)

    print(model.summary())

    bool_mask = K.greater_equal(orig, x)
    mask = K.cast(bool_mask, dtype='float32')

    mask_input = Input(tensor=mask)  # Makes the mask to a Keras tensor to use as input
    x = keras.layers.multiply([mask_input, x])
    x = Conv2D(512, (3, 3), padding='same')(x)
    model = Model(inputs=img_input, outputs=x)

    print(model.summary())

    x = Reshape((n_classes, w_image * h_image))(x)
    x = Permute((2, 1))(x)
    main_output = Activation('softmax')(x)

    model = Model(inputs=img_input, outputs=main_output)

    print(model.summary())
