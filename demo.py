import os
import random

import cv2 as cv
import keras.backend as K
import numpy as np

from model import create_model

if __name__ == '__main__':
    img_rows, img_cols = 320, 320
    channel = 4

    model_weights_path = 'models/model.44-0.0239.hdf5'
    model = create_model()

    model.load_weights(model_weights_path)
    print(model.summary())

    test_path = 'data/test/'
    test_images = [f for f in os.listdir(test_path) if
                   os.path.isfile(os.path.join(test_path, f)) and f.endswith('.jpg')]

    samples = random.sample(test_images, 10)

    for i in range(len(samples)):
        image_name = samples[i]
        filename = os.path.join(test_path, image_name)

        print('Start processing image: {}'.format(filename))

        x_test = np.empty((1, img_rows, img_cols, 4), dtype=np.float32)
        bgr_img = cv.imread(filename)
        gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
        rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
        rgb_img = rgb_img / 255.
        x_test[0, :, :, 0:3] = rgb_img
        x_test[0, :, :, 3] = np.random.uniform(0, 1, (img_rows, img_cols))

        out = model.predict(x_test)
        # print(out.shape)

        out = np.reshape(out, (img_rows, img_cols))
        out = out * 255.0
        out = out.astype(np.uint8)

        cv.imwrite('images/{}_image.png'.format(i), bgr_img)
        cv.imwrite('images/{}_out.png'.format(i), out)
        cv.imwrite('images/{}_gray.png'.format(i), gray_img)

    K.clear_session()
