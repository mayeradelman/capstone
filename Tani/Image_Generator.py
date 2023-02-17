import cv2
import numpy as np
from PIL import Image, ImageChops
from skimage.util import random_noise
from skimage.util import img_as_float


def add_noise(img, amount = .2, salt_vs_pepper = .95):
    im_float = img_as_float(img)
    im_noise = random_noise(im_float, mode='s&p', amount=.2,salt_vs_pepper=.95)
    im_noise = Image.fromarray((im_noise * 255).astype(np.uint8))
    # change the image to black and white
    im_noise = im_noise.convert('L')
    return im_noise

def image_affine(image, intensity = 30):
    width, height = image.size
    width, height = image.size
    
    # inverts the image. This is done because the background is white and the text is black
    # and editing the image with the background as black and the text as white is easier
    im = ImageChops.invert(image)
   
    # turn im into cv2 image in order to use cv2 functions
    im = np.array(im)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    
    # these three points represent three corners of the image (top left, top right, bottom left)
    pt1 = np.float32([[0,0],[width,0],[0,height]])
    
    # choose random points based on the intensity parameter
    pt2 = np.float32([[np.random.randint(0, intensity),np.random.randint(0, intensity)],
                      [np.random.randint(width-intensity, width),np.random.randint(0, intensity)],
                      [np.random.randint(0, intensity),np.random.randint(height-intensity, height)]])
    
    M = cv2.getAffineTransform(pt1, pt2)
    rows, cols, ch = im.shape
    dst = cv2.warpAffine(im, M, (cols, rows))

    
    
    return cv2.bitwise_not(dst)

def image_generator(image, im_num = 1, intensity = 30, noise = True, affine = True, noise_amount = .2, salt_vs_pepper = .95):
    # im_num is the number of images you want to generate
    # intensity is the intensity of the affine transformation
    # noise is a boolean that determines whether or not to add noise to the image
    # affine is a boolean that determines whether or not to apply an affine transformation to the image
    images = []
    for i in range(im_num):
        if affine == True:
            image = image_affine(image, intensity)
        if noise == True:
            image = add_noise(image, noise_amount, salt_vs_pepper)
        images.append(image)

    return images