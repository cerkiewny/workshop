import cv2 as cv
from PIL import Image
import sys
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread(sys.argv[1], 0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

Image.fromarray(magnitude_spectrum, 'RGB').save(sys.argv[2])
