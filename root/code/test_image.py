import cv2

from root.code.cnn import CNN

cnn = CNN(100)
cnn.model_execution(.1, 32, 16, 3, 2, 1)
cnn.model_save()
