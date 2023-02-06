# The purpose of this code is to find all the letters from the training and testing
#  data from within their respective folders, and to combine them all and turn them
#  into a usable 28x28 image.
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# The code assumes it is run from within the same folder as hhd_dataset_cleaned
train_directory = '.\hhd_dataset_cleaned\TRAIN'
test_directory = '.\hhd_dataset_cleaned\TEST'
 
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
            # Convert the image to a numpy array and resize the image and 
            # convert it to grayscale (this removes the third dimension)
            image = np.array(Image.open(f).convert('L').resize((28, 28)))
            # Add the image to the X list
            X_data.append(image)
            # Split the directory in order to get the file name
            strings = f.split("\\")
            file_ending = strings[len(strings)-1]
            # Split the file name in order to get the ID number
            letter_number = file_ending.split('_')[0]
            # Add the ID number to the y list
            y_data.append(int(letter_number))
    return X_data, y_data

X_train, y_train = get_data(train_directory)

print(len(X_train))
print(len(y_train))

X_test, y_test = get_data(test_directory)

print(len(X_test))
print(len(y_test))

X_test = np.array(X_test)
y_test = np.array(y_test)

fig, ax = plt.subplots(nrows=3, ncols=9, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(27):
  img = X_test[y_test == i][0].reshape(28, 28)
  ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()