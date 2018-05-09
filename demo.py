import argparse

import cv2 as cv
import keras.backend as K
import numpy as np

from new_start import autoencoder

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4

    model_weights_path = 'models/model.95-0.05.hdf5'
    model = autoencoder(img_rows, img_cols, channel)
    model.load_weights(model_weights_path)
    print(model.summary())

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file")
    args = vars(ap.parse_args())
    filename = args["image"]
    if filename is None:
        car_ids = ['00133', '06258', '07602']
        for car_id in car_ids:
            filename = 'images/{}.jpg'.format(car_id)

            print('Start processing image: {}'.format(filename))

            x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
            bgr_img = cv.imread(filename)
            gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
            rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
            rgb_img = rgb_img / 255.
            x_test[0, :, :, 0:3] = rgb_img
            x_test[0, :, :, 3] = np.random.uniform(0, 1, (img_rows, img_cols))
            # rgb_img = np.expand_dims(rgb_img, 0)

            rep = model.predict(x_test)
            print(rep.shape)

            rep = np.reshape(rep, (img_rows, img_cols))
            rep = rep * 255.0
            rep = rep.astype(np.uint8)
            print(rep)
            cv.imshow('rep', rep)
            cv.imwrite('images/{}_out.jpg'.format(car_id), rep)
            cv.imwrite('images/{}_gray.jpg'.format(car_id), gray_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    K.clear_session()
