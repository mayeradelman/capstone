# The purpose of this code is to find all the letters from the training and testing
#  data from within their respective folders, and to combine them all and turn them
#  into a usable 28x28 image.
import os
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# The code assumes it is run from within the same folder as filled_in
train_directory = '.\\filled_in\TRAIN'
# largest_length = 0
# largest_width = 0

def process_image(image, y):
    length = image.size[1]
    length_diff = 350 - length
    width = image.size[0]
    width_diff = 350 - width
    # global largest_length
    # global largest_width
    # if length > largest_length:
    #     largest_length = length
    # if width > largest_width:
    #             largest_width = width
    grayscale_image = image.convert('L')
    inverted_image = ImageOps.invert(grayscale_image)
    bordered_image = inverted_image.crop((0-width_diff/2., 0-length_diff/2., width+width_diff/2., length+length_diff/2.))
    np_image = np.array(bordered_image.resize((28,28)))
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
            y_data.append(int(letter_number))

            # Convert the image to a numpy array and resize the image and 
            # convert it to grayscale (this removes the third dimension)
            image = process_image(Image.open(f), int(letter_number))
            # Add the image to the X list
            X_data.append(image)
    return X_data, y_data

X_train, y_train = get_data(train_directory)

print(len(X_train))
print(len(y_train))

X_train = np.array(X_train)
y_train = np.array(y_train)

X_reshaped = np.array([i.flatten() for i in X_train])

np.save('X_Data.npy', X_reshaped)
np.save('Y_Data.npy', y_train)

# Largest length is 350 and largest width is 325
# print(largest_length)
# print(largest_width)

fig, ax = plt.subplots(nrows=3, ncols=9, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(27):
  img = X_train[y_train == i][0]#.reshape(28, 28)
  ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()