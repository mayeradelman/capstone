import cv2
import numpy as np
from PIL import Image
import os


def show_before_after(img_before, img_after):
    # display the result
    cv2.imshow("Before", img_before)
    cv2.waitKey(2000)
    cv2.imshow("After", img_after)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

def remove_yellow_pixels(img_path, new_img_path):
    # Open the image file
    im = Image.open(img_path)
    # im.show("Before Removing Yellow")

    # Convert the image to RGB mode
    im = im.convert("RGB")

    # Get the pixel data
    pixels = im.load()

    # Define the color as yellow
    color_to_remove = (255, 255, 0)

    # color distance threshold
    threshold = 255

    # Define the color distance function
    def color_distance(c1, c2):
        return sum([(x-y)**2 for x, y in zip(c1, c2)])**0.5

    # Iterate through all pixels
    for x in range(im.width):
        for y in range(im.height):
            # Check if the color distance between the pixel color and the color to remove is less than the threshold
            if color_distance(pixels[x, y], color_to_remove) < threshold:
                # print("changing color")
                # Set the pixel color to white 
                pixels[x, y] = (255, 255, 255)

    # Save the modified image
    directory = os.path.dirname(new_img_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # im.show("After Removing Yellow")
    im.save(new_img_path)

def fill_in_image(img_path, new_image_path):
    # Load the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("Before Fill", img)
    # cv2.waitKey(1000)
    # Invert the image
    img = cv2.bitwise_not(img)

    # Define the structuring element and perform morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Invert the image back
    img = cv2.bitwise_not(img)

    directory = os.path.dirname(new_image_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # cv2.imshow("After Fill", img)
    # cv2.waitKey(1500)
    cv2.imwrite(new_image_path, img)

img_before = cv2.imread(r"C:\Users\emwil\OneDrive\Documents\Capstone\hhd_dataset_cleaned\TRAIN\0\0_84.png")
img_after = cv2.imread(r"C:\Users\emwil\OneDrive\Documents\Capstone\filled_in\TRAIN\0\0_84.png")
show_before_after(img_before, img_after)

# iterate through all image files and fix the images

# for root, dirs, files in os.walk("hhd_dataset_cleaned"):
#     for filename in files:
#         if filename.endswith(".png"):
#             original_img = "C:\\Users\\emwil\\OneDrive\\Documents\\Capstone\\" + os.path.join(root, filename)
#             without_yellow_path = os.path.join(root, filename).replace("hhd_dataset_cleaned", "C:\\Users\\emwil\\OneDrive\\Documents\\Capstone\\without_yellow_streaks")
#             filled_in_path = os.path.join(root, filename).replace("hhd_dataset_cleaned", "C:\\Users\\emwil\\OneDrive\\Documents\\Capstone\\filled_in")
#             remove_yellow_pixels(original_img, without_yellow_path)
#             fill_in_image(without_yellow_path, filled_in_path)
            