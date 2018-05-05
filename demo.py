from autoencoder import autoencoder

if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    channel = 3

    model_weights_path = 'models/model.362-0.06.hdf5'
    model = autoencoder(img_rows, img_cols, channel)
    model.load_weights(model_weights_path)

