from matplotlib import pyplot as plt
import numpy as np
import cv2
import tools


def getPixel(x,y,file):
    img = cv2.imread(file)
    return img[x,y]


def getPixelsColor(r,g,b,file):
    img = cv2.imread(file)
    return img[r,g,b]

def getImage(x1,y1,x2,y2,img):
    return img[x1:y1,x2:y2]
