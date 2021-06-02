
import os
import pandas as pd
import shutil
from cv2 import cv2
import face_recognition as fr
import numpy as np
import random
from instafilter import Instafilter

def main():

    image_path = 'original_censored.png'

    # Resize 145x145.
    img = cv2.imread(image_path)   # Loaded as RGBA image.
    img_145 = cv2.resize(img, (145, 145), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite('original_145x145.png', img_145)

    # Resize 64x64.
    img_64 = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite('original_64x64.png', img_64)

    # Landmarks
    img = fr.load_image_file(image_path)
    faces_landmarks = fr.face_landmarks(img)
    top, right, bottom, left = 0, img.shape[0], img.shape[1], 0
    landmarks = fr.face_landmarks(img, face_locations=[[top, right, bottom, left]])[0]

    # Landmarks image.
    img = cv2.imread(image_path)
    for k in landmarks:
        for p in landmarks[k]:
            landmarks_image = cv2.circle(img, p, 1, (0, 0, 255), 1)
    cv2.imwrite('landmark_image.png', img)

    # Filters.
    img = cv2.imread(image_path)

    # Dog nose.
    new_image = dog_nose(img, landmarks, 'filter_images/dog_nose.png')
    cv2.imwrite('dog_nose.png', new_image)

    # Glasses (see-through).
    new_image = glasses(img, landmarks, 'filter_images/glasses.png')
    cv2.imwrite('glasses.png', new_image)

    # Glasses - Shades (very limited see-through). 95% opacity.
    new_image = glasses(img, landmarks, 'filter_images/shades_leak.png')
    cv2.imwrite('shades_leak.png', new_image)

    # Glasses - Shades 100% opacity.
    new_image = glasses(img, landmarks, 'filter_images/shades_no_leak.png')
    cv2.imwrite('shades_no_leak.png', new_image)

    # -------------------- Color Filters [Instagram]. -------------------- #

    top_filter_names = [
        'Slumber', 'Skyline', 'Dogpatch', 'Aden', 
        'Valencia', 'Ludwig', 'Gingham', 'Hudson', 'Ashby']

    for filter_name in top_filter_names:
        new_image = apply_instagram_filter(image_path, filter_name)
        cv2.imwrite(filter_name + '.png', new_image)




def apply_instagram_filter(image_name, filter_name):
    """
    Input: Full path and image name of image, output the filtered image.
    """
    model = Instafilter(filter_name)
    new_image = model(image_name)
    return new_image

def mustasch(img_orig, landmarks, img_filter, scale_p=200):
    """
    Applies mustasch to the image. filter image starts at scale 200% increased scale from the found area of the upper lip by facial landmarks. If the image is too big it is reduced by 5% until it is small enough to fit.
    """
    while True:
        try:

            img = img_orig # Keep original clean for recursive call - Reduction of nose.

            # Make img RGBA.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

            # Load mustasch.
            filter_img = cv2.imread(img_filter, -1)   # Loaded as RGBA image.
            top_lip = landmarks['top_lip'] # Width
            nose_tip = landmarks['nose_tip']    # Top height, with slight offset.    
            
            xvalues_top_lip = [x for x, y in top_lip] # x1, x2
            yvalues_top_lip = [y for x, y in top_lip] # y1
            yvalues_nose_tip = [y for x, y in nose_tip] # y2
            
            # Upper lip region area.
            filter_region_width = max(xvalues_top_lip) - min(xvalues_top_lip)
            filter_region_height = abs(max(yvalues_nose_tip) - max(yvalues_top_lip)) # Image Y lim 0 => up, y lim inf => down.

            scale_percent = scale_p  # Percent of original size. The target area of the face.
            new_filter_region_width = int(filter_region_width * scale_percent / 100)
            new_filter_region_height = int(filter_region_height * scale_percent / 100)
            filter_img = cv2.resize(filter_img, (new_filter_region_width, new_filter_region_height), interpolation=cv2.INTER_LANCZOS4)

            # Area to to place the filter image on the img.
            center_pos = int(max(yvalues_nose_tip) + filter_region_height//2), int(np.median(xvalues_top_lip))   # REVERSE WORKS ???

            w1, w2 = integer_even_division(filter_img.shape[0])   # Split nose_img width into 2 integers.
            h1, h2 = integer_even_division(filter_img.shape[1])   # Split nose_img height into 2 integers.

            x1, x2 = center_pos[0] - w1, center_pos[0] + w2   # Left, right pos.
            y1, y2 = center_pos[1] - h1, center_pos[1] + h2   # Down, up pos.

            # Add nose_image to img, with transparent background.
            alpha_s = filter_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                img[x1:x2, y1:y2, c] = (alpha_s * filter_img[:, :, c] + alpha_l * img[x1:x2, y1:y2, c])

            return img

        except:
            scale_p -= 5    # Reduce scale 5% each time its found too big.

def glasses(img_orig, landmarks, img_filter, scale_p=135):
    """
    Applies glasses to the image. filter image starts at scale 125% increased scale from the found area of the eyes by facial landmarks. If the image is too big it is reduced by 5% until it is small enough to fit.
    """

    while True:
        try:

            img = img_orig # Keep original clean for recursive call - Reduction of nose.

            # Make img RGBA.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

            # Load glasses.
            filter_img = cv2.imread(img_filter, -1)   # Loaded as RGBA image.
            left_eyebrow = landmarks['left_eyebrow']    # For glasses width.
            right_eyebrow = landmarks['right_eyebrow']  # For glasses width.
            nose_bridge = landmarks['nose_bridge']      # For glasses height.
            xvalues_left_eyebrow = [x for x, y in left_eyebrow] # x1
            yvalues_left_eyebrow = [y for x, y in left_eyebrow] # y2
            xvalues_right_eyebrow = [x for x, y in right_eyebrow]   # x2
            xvalues_nose_bridge = [x for x, y in nose_bridge]   # center
            yvalues_nose_bridge = [y for x, y in nose_bridge]   # y1 

            # Eye region area.
            filter_region_width = max(xvalues_right_eyebrow) - min(xvalues_left_eyebrow)
            filter_region_height = abs(min(yvalues_left_eyebrow) - int(np.percentile(yvalues_nose_bridge, 75))) # Image Y lim 0 => up, y lim inf => down.

            scale_percent = scale_p  # Percent of original size. The target area of the face.
            new_filter_region_width = int(filter_region_width * scale_percent / 100)
            new_filter_region_height = int(filter_region_height * scale_percent / 100)
            filter_img = cv2.resize(filter_img, (new_filter_region_width, new_filter_region_height), interpolation=cv2.INTER_LANCZOS4)

            # Area to to place the filter image on the img.
            center_pos = int(np.median(yvalues_nose_bridge)), int(np.median(xvalues_nose_bridge))   # REVERSE WORKS ???

            w1, w2 = integer_even_division(filter_img.shape[0])   # Split nose_img width into 2 integers.
            h1, h2 = integer_even_division(filter_img.shape[1])   # Split nose_img height into 2 integers.

            x1, x2 = center_pos[0] - w1, center_pos[0] + w2   # Left, right pos.
            y1, y2 = center_pos[1] - h1, center_pos[1] + h2   # Down, up pos.

            # Add nose_image to img, with transparent background.
            alpha_s = filter_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                img[x1:x2, y1:y2, c] = (alpha_s * filter_img[:, :, c] + alpha_l * img[x1:x2, y1:y2, c])

            return img

        except:
            scale_p -= 5    # Reduce scale 5% each time its found too big.

def dog_nose(img_original, landmarks, img_filter, scale_p=300):
    """
    Applies a dog nose to the image. Nose starts at scale 200% increased scale from the found area of the nose by facial landmarks. If the image is too big it is reduced by 5% until it is small enough to fit.
    """
    while True:
        try:
            img = img_original # Keep original clean for recursive call - Reduction of nose.

            # Make img RGBA.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

            # Load dog nose.
            nose_img = cv2.imread(img_filter, -1) # Load as RGBA.   
            nose_tip = landmarks['nose_tip']
            nose_bridge = landmarks['nose_bridge']
            xvalues_nose_tip = [x for y, x in nose_tip]  # nose_tip.    [For some reason landmarks output is reverse]
            yvalues_nose_tip = [y for y, x in nose_tip]  # nose_tip.    [For some reason landmarks output is reverse]
            xvalues_nose_bridge = [x for y, x in nose_bridge]   # nose_bridge.  [For some reason landmarks output is reverse]

            # Resize nose_img.
            nose_width = max(yvalues_nose_tip) - min(yvalues_nose_tip)
            nose_height = max(xvalues_nose_bridge) - min(xvalues_nose_bridge)

            scale_percent = scale_p  # Percent of original size. The target area of the face.
            new_nose_width = int(nose_width * scale_percent / 100)
            new_nose_height = int(nose_height * scale_percent / 100)
            nose_img = cv2.resize(nose_img, (new_nose_width, new_nose_height), interpolation=cv2.INTER_LANCZOS4)

            # Nose area - Area to be replaced with 2nd image.
            center_nose_pos = int(np.median(xvalues_nose_tip)), int(np.median(yvalues_nose_tip))
        
            w1, w2 = integer_even_division(nose_img.shape[0])   # Split nose_img width into 2 integers.
            h1, h2 = integer_even_division(nose_img.shape[1])   # Split nose_img height into 2 integers.

            x1, x2 = center_nose_pos[0] - w1, center_nose_pos[0] + w2   # Left, right pos.
            y1, y2 = center_nose_pos[1] - h1, center_nose_pos[1] + h2   # Down, up pos.

            # Add nose_image to img, with transparent background.
            alpha_s = nose_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                img[x1:x2, y1:y2, c] = (alpha_s * nose_img[:, :, c] + alpha_l * img[x1:x2, y1:y2, c])

            return img

        except:
            scale_p -= 5    # Reduce scale 5% each time its found too big.

def integer_even_division(i):
    """
    Division of an integer into 2 even integers.
    """
    i1 = i // 2
    i2 = i - i1
    return i1, i2



if __name__ == '__main__':
    main()