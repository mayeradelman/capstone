# The purpose of this code is to find all the letters from the training and testing
#  data from within their respective folders, and to combine them all and turn them
#  into a usable 28x28 image.
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from Image_Cleaner import *
from Image_Generator import *

# largest_length = 0
# largest_width = 0

def process_image(image, y):
    # Clean image from yellow marks
    image = remove_yellow(image)

    # Crop the image so all images are proportional
    length = image.size[1]
    length_diff = 350 - length
    width = image.size[0]
    width_diff = 350 - width
    grayscale_image = image.convert('L')
    inverted_image = ImageOps.invert(grayscale_image)
    bordered_image = inverted_image.crop((0-width_diff/2., 0-length_diff/2., width+width_diff/2., length+length_diff/2.))

    # Resize the image and convert to a NumPy array
    np_image = np.array(bordered_image.resize((180,180)))
    return np_image

# This function finds each image and adds it to the X list, and finds the correct
# ID number from the image name and adds it to the y list
def get_data(directory):
    X_data = []
    y_data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # If this next item is a directory, dig deeper
        if os.path.isdir(f):
            temp_X_data, temp_y_data = get_data(f)
            for i in temp_X_data:
                X_data.append(i)
            for i in temp_y_data:
                y_data.append(i)
        # If this next item is a file, it is one of our images 
        elif os.path.isfile(f):
            # Split the directory in order to get the file name
            strings = f.split("\\")
            file_ending = strings[len(strings)-1]
            # Split the file name in order to get the ID number
            letter_number = file_ending.split('_')[0]
            # Add the ID number to the y list
            id_tag = int(letter_number)
            y_data.append(id_tag)

            # Convert the image to a numpy array and resize the image and 
            # convert it to grayscale (this removes the third dimension)
            pil_image = Image.open(f)
            image = process_image(pil_image, id_tag)
            # Add the image to the X list
            X_data.append(image)

            # Add generated images from this image to the X and y list
            generated_images = image_generator(pil_image, im_num=2)
            for new_image in generated_images:
                new_image = process_image(new_image, id_tag)
                X_data.append(new_image)
                y_data.append(id_tag)
    return X_data, y_data

# The code assumes it is run from within the same folder as hhd_cleaned+merged
train_directory = '.\\hhd_cleaned+merged\TRAIN'

X_train, y_train = get_data(train_directory)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(len(X_train))
print(len(y_train))

X_reshaped = np.array([i.flatten() for i in X_train])

np.save('X_Data.npy', X_reshaped)
np.save('Y_Data.npy', y_train)

# Largest length is 350 and largest width is 325
# print(largest_length)
# print(largest_width)

fig, ax = plt.subplots(nrows=3, ncols=9, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(27):
  img = X_train[y_train == i][6]#.reshape(28, 28)
  ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()