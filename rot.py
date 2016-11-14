import cv2
import imutils
import numpy as np

img = cv2.imread('002.jpg')

num_rows, num_cols = img.shape[:2]

rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 1)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))


cv2.imshow('ori2', img_rotation)
cv2.imwrite('003.png',img_rotation)
