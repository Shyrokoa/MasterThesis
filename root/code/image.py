import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from root.code.cnn import check


class Image:

    def __init__(self, path, size, model):
        self.model = model
        self.size = size
        self.img_size = (20, 20)
        self.path = path
        self.img = ''
        self.read_image()
        self.im_crop()
        self.im_contrast()
        self.im_threshold(130)
        self.im_blur(4)
        self.im_threshold(175)
        self.im_blur(4)
        self.im_threshold(160)
        self.rgb_2_gray()
        self.detect_objects()

    def read_image(self):
        self.img = cv2.imread(self.path, 1)

    def im_show(self):
        plt.figure(figsize=self.img_size)
        plt.imshow(self.img, cmap='gray')
        plt.show()

    def im_crop(self):
        self.img = self.img[180:557, 417:795]
        self.img = cv2.resize(self.img, (500, 500), interpolation=cv2.INTER_AREA)
        #self.im_show()

    def im_contrast(self):
        alpha = 2  # Contrast control (1.0-3.0)
        beta = 50  # Brightness control (0-100)
        self.img = cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)
        #self.im_show()

    def im_threshold(self, limit):
        # Threshold on the crop and contrast image
        _, self.img = cv2.threshold(self.img, limit, 255, cv2.THRESH_BINARY)
        #self.im_show()

    def im_blur(self, size):
        self.img = cv2.blur(self.img, (size, size))
        #self.im_show()

    def rgb_2_gray(self):
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def detect_objects(self):
        contours, hierarchy = cv2.findContours(self.img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        self.img = cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        coeff = 1.53506295623346405
        contours, _ = cv2.findContours(self.img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)

        path = r'C:\Users\SHYROKOA\Desktop\test'
        dest = r'C:\Users\SHYROKOA\Desktop\blur'

        for i, contour in enumerate(contours):
            if i != len(contours) - 1:   # we don't need the last contour
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w/h
                crop_img = self.img[y:y+h, x:x+w]

                # calculate the ox and oy
                ox = x + w/2
                oy = y + h/2

                # calculate the black pixels number
                black = count_black(crop_img)

                # calculate the white pixels number
                white = black / coeff

                # total box pixels
                total = white + black

                b = int(round(math.sqrt(total / aspect_ratio)))
                a = int(round(aspect_ratio * b))

                # new coordinates
                x = int(round(ox - a/2))
                y = int(round(oy - b/2))
                w = a
                h = b

                crop_img = cv2.resize(crop_img, (50, 50))
                crop_img = cv2.blur(crop_img, (3, 3))
                _, crop_img = cv2.threshold(crop_img, 160, 255, cv2.THRESH_BINARY)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
                qw = check(crop_img, self.size, self.model)

                if qw == 'circle':
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if x < 250:
                        cv2.putText(self.img, f'Circle Detected {i} -> X: {(x + w) / 2}, y: {(y + h) / 2}',
                                    (x + w + 10, y + h - 5), 0, 0.26, (192, 52, 52))
                    else:
                        cv2.putText(self.img, f'Circle Detected {i} -> X: {(x + w) / 2}, y: {(y + h) / 2}',
                                    (x + w + 10 - 200, y + h - 5), 0, 0.26, (192, 52, 52))
                else:
                    cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    if x < 250:
                        cv2.putText(self.img, f'Not circle Detected {i} -> X: {(x + w) / 2}, y: {(y + h) / 2}',
                                    (x + w + 10, y + h - 5), 0, 0.26, (192, 52, 52))
                    else:
                        cv2.putText(self.img, f'Not circle Detected {i} -> X: {(x + w) / 2}, y: {(y + h) / 2}',
                                    (x + w + 10 - 200, y + h - 5), 0, 0.26, (192, 52, 52))
        self.im_show()


def count_black(img):
    cnt = ''
    colors, counts = np.unique(img.reshape(-1, 1), axis=0, return_counts=True)
    for color, count in zip(colors, counts):
        if color[0] == 0:
            cnt = count
    return cnt


