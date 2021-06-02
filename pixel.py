from PIL import Image
import matplotlib.pyplot as plt
import os

def pixelate(img, dims):
    imgSmall = img.resize((dims,dims))
    result = imgSmall.resize(img.size,Image.NEAREST)
    result.save(f[:-4]+'_censored.png')
    return result


all_files = os.listdir()
for f in all_files:
    # Only image files (.png).
    if '.png' in f and '_censored' not in f:
        img = Image.open(f)
        pixel_image = pixelate(img, 30)

