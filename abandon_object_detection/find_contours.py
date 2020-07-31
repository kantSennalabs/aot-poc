import cv2 
import numpy as np


# while True:
min_w=15
min_h=15 
max_w=500 
max_h=500
im = cv2.imread('contours.png')
im = cv2.resize(im, (400,400))
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
arr = cv2.dilate(imgray, kernel, iterations=2)
arr = np.array(arr, dtype=np.uint8) 
_, th = cv2.threshold(arr,127,255,0)
cv2.imshow('th',th)
cv2.imshow('arr',arr)

contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if (w >= 50) and (h >= 50):
        cv2.rectangle(im, (x,y), (x+w,y+h), (0,0,255), 2)

cv2.imshow('rst',im)
# cv2.imshow('th',th)
cv2.waitKey(0)