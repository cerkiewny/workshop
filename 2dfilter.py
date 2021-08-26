import numpy as np
import cv2 as cv
import sys
from matplotlib import pyplot as plt

img = cv.imread(sys.argv[1], 0)
kernel = np.array([
[-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1],
[-1,-1, 25,-1,-1],
[-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1]
], np.float32)

#kernel = np.array([
#[-4,-4,-4,-4,-4],
#[-4,7,7,7,-4],
#[-4,7,15,7,-4],
#[-4,7,7,7,-4],
#[-4,-4,-4,-4,-4]
#], np.float32)
#
#kernel = np.array([
#    [1, 2, 1],
#    [2, 4, 2],
#    [1, 2, 1],
#], np.float32)

#kernel = np.array([
#[1,2,4,2,1],
#[2,4,8,4,1],
#[4,8,16,8,1],
#[2,4,1,4,1],
#[1,2,4,2,1]
#], np.float32)
#kernel = np.array([
#    [-1, -1, -1],
#    [-1, 8, -1],
#    [-1, -1, -1],
#], np.float32)


if np.sum(kernel) != 0:
    kernel /= np.sum(kernel)

dst = cv.filter2D(img,-1,kernel)
#for i in range(30):
#    dst = cv.filter2D(dst,-1,kernel)

plt.subplot(121),plt.imshow(img, cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst, cmap='gray'),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
