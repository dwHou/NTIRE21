#!/usr/bin/env python

import torch
import cv2
from PIL import Image

img = Image.open('test.png').convert('RGB')
print(img.size)
print(img.getpixel((0,0)))

img_yuv = Image.open('test.png').convert("YCbCr")
print(img_yuv.size)
print(img_yuv.getpixel((0,0)))


img_re = img_yuv.convert('RGB')
print(img_re.size)
print(img_re.getpixel((0,0)))

img = cv2.imread('test.png')
print(img[0, 0, :])
img_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
print(img_yuv[0, 0, :])
img_ycrbr = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
print(img_ycrbr[0, 0, :])
img_re = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR)
print(img_re[0, 0, :])

