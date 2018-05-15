import cv2 as cv
import keras.backend as K
import numpy as np

from utils import custom_loss

if __name__ == '__main__':
    file_id = '07647'
    filename = 'images/samples/{}.jpg'.format(file_id)
    bgr_img = cv.imread(filename)
    y_true = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    cv.imwrite( 'images/samples/{}_gray.jpg'.format(file_id), y_true)
    y_true = y_true / 255.
    y_true = y_true.astype(np.float32)

    filename = 'images/samples/{}_out.jpg'.format(file_id)
    y_pred = cv.imread(filename, 0)
    y_pred = y_pred / 255.
    y_pred = y_pred.astype(np.float32)
    loss = custom_loss(y_true, y_pred)
    print(K.eval(loss))
