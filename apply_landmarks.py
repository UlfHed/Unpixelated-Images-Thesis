import face_recognition as fr
from cv2 import cv2 


image_path = 'original.png'

# Landmarks
img = fr.load_image_file(image_path)
faces_landmarks = fr.face_landmarks(img)
top, right, bottom, left = 0, img.shape[0], img.shape[1], 0
landmarks = fr.face_landmarks(img, face_locations=[[top, right, bottom, left]])[0]

# Landmarks image.
img = cv2.imread(image_path)
for k in landmarks:
    for p in landmarks[k]:
        landmarks_image = cv2.circle(img, p, 1, (0, 0, 255), 2)
cv2.imwrite('landmark_image.png', img)

# Censor landmark image.

from PIL import Image
import matplotlib.pyplot as plt
import os

def pixelate(img, dims):
    imgSmall = img.resize((dims,dims))
    result = imgSmall.resize(img.size,Image.NEAREST)
    result.save('landmark_image.png'[:-4]+'_censored.png')
    return result


img = Image.open('landmark_image.png')
pixel_image = pixelate(img, 30)

