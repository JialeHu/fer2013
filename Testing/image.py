import numpy as np
import os
import sys
import cv2
import math
from matplotlib import pyplot as plt

data = np.genfromtxt('/Users/hkh/Desktop/FR/fer2013.csv',delimiter=',',dtype=None,encoding=None)
print(data[0])
print(type(data[1,0]))
a = np.double(data[1,0])
print(a)
b = np.fromstring(data[1,1], dtype=int, sep=' ')
print(type(b))
print(b.shape[0])
im = np.reshape(b,(int(math.sqrt(b.shape[0])),int(math.sqrt(b.shape[0]))))
print(im.shape)
imGray = np.zeros(im.shape, np.double)
cv2.normalize(im,imGray, 1.0, 0.0, cv2.NORM_MINMAX, dtype=cv2.CV_64F)
cv2.imshow('image',imGray)
cv2.waitKey(0)
cv2.destroyAllWindows()