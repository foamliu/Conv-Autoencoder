import argparse

import cv2 as cv
import numpy as np

from model import create_model

if __name__ == '__main__':
    img_rows, img_cols = 224, 224
    channel = 3

    model_weights_path = 'models/model.362-0.06.hdf5'
    model = create_model()
    model.load_weights(model_weights_path)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file")
    args = vars(ap.parse_args())

    filename = args["image"]
    if filename is None:
        filename = 'images/samples/05509.jpg'

    bgr_img = cv.imread(filename)
    rgb_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2RGB)
    rgb_img = rgb_img / 255.
    rgb_img = np.expand_dims(rgb_img, 0)

    rep = model.predict(rgb_img)
    print(rep.shape)

    rep = np.reshape(rep, (224, 224))
    rep = rep * 255.0
    rep = rep.astype(np.uint8)
    print(rep)
    cv.imshow('rep', rep)
    cv.imwrite('out.jpg', rep)
    cv.waitKey(0)
    cv.destroyAllWindows()
