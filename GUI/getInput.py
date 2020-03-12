import cv2
from PIL import Image
import PIL.ImageOps
import numpy as np
from pylab import *
def getContour(name, low, high):
    img = cv2.imread(name, 0)
    canny = cv2.Canny(img, low, high)
    cv2.imwrite('F:\\test\\data\\contour.png', canny)
    image = Image.open('F:\\test\\data\\contour.png')
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('F:\\test\\data\\contour.png')


def getGradient(name):
    gray = plt.imread('F:\\test\\data\\contour.png')
    img = plt.imread(name)
    size = tuple(gray.shape)
    # print(gray)
    width = size[0]
    height = size[1]
    Rx = np.zeros((width, height, 3), dtype=np.double)
    Ry = np.zeros((width, height, 3), dtype=np.double)
    for i in range(width-1):
        for j in range(height - 1):
            for k in range(3):
                if (gray[i, j] != 1):
                    Rx[i, j, k] = img[i+1, j, k] - img[i, j, k]
                    Ry[i, j, k] = img[i, j+1, k] - img[i, j, k]
    grad = Rx + Ry
    # grad = cv2.convertScaleAbs(grad)
    plt.imsave('F:\\test\\data\\gradient.png', grad)

def getColor(name):
    gray = plt.imread('F:\\test\\data\\contour.png')
    img = plt.imread(name)
    size = tuple(gray.shape)
    # print(gray)
    width = size[0]
    height = size[1]
    R = np.zeros((width, height, 3), dtype=np.double)
    for i in range(width):
        for j in range(height-1):
            for k in range(3):
                if (gray[i, j] != 1):
                    if j == 1 :
                        R[i, j+1, k] = img[i, j+1, k]
                    else:
                        R[i, j - 1, k] = img[i, j - 1, k]
                        R[i, j + 1, k] = img[i, j + 1, k]
    plt.imsave('F:\\test\\data\\color.png', R)