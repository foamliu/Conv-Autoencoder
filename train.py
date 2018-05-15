import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import migrate
from model import create_model
from utils import load_data

if __name__ == '__main__':
    batch_size = 16
    epochs = 1000
    patience = 50

    # Load our model
    model = create_model()
    migrate.migrate_model(model)

    print(model.summary())

    # Load our data
    x_train, y_train, x_valid, y_valid = load_data()

    # Callbacks
    tensor_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    trained_models_path = 'models/model'
    model_names = trained_models_path + '.{epoch:02d}-{val_loss:.4f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, monitor='val_loss', verbose=1, save_best_only=True)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
    callbacks = [tensor_board, model_checkpoint, early_stop, reduce_lr]

    # Start Fine-tuning
    model.fit(x_train,
              y_train,
              validation_data=(x_valid, y_valid),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks,
              verbose=1
              )
