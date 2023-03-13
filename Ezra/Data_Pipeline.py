# The purpose of this code is to find all the letters from the training and testing
#  data from within their respective folders, and to combine them all and turn them
#  into a usable 180x180 (unless otherwise specified) image.
import os
import numpy as np
from PIL import Image, ImageOps
from Image_Cleaner import *
from Image_Generator import *

import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from collections import defaultdict

# largest_length = 0
# largest_width = 0

def process_image(image, y, image_size):
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
    np_image = np.array(bordered_image.resize((image_size, image_size)))
    return np_image

# This function finds each image and adds it to the X list, and finds the correct
# ID number from the image name and adds it to the y list
def get_data(directory, image_size, image_number):
    X_data = []
    y_data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # If this next item is a directory, dig deeper
        if os.path.isdir(f):
            temp_X_data, temp_y_data = get_data(f, image_size, image_number)
            for i in temp_X_data:
                X_data.append(i)
            for i in temp_y_data:
                y_data.append(i)
        # If this next item is a file, it is one of our images 
        elif os.path.isfile(f):
            # Split the directory in order to get the file name
            strings = f.split(os.sep)
            file_ending = strings[len(strings)-1]
            # Split the file name in order to get the ID number
            letter_number = file_ending.split('_')[0]
            # Add the ID number to the y list
            id_tag = int(letter_number)
            y_data.append(id_tag)

            # Convert the image to a numpy array and resize the image and 
            # convert it to grayscale (this removes the third dimension)
            pil_image = Image.open(f)
            image = process_image(pil_image, id_tag, image_size)
            # Add the image to the X list
            X_data.append(image)

            # Add generated images from this image to the X and y list
            generated_images = image_generator(pil_image, im_num=image_number)
            for new_image in generated_images:
                new_image = process_image(new_image, id_tag, image_size)
                X_data.append(new_image)
                y_data.append(id_tag)
    return X_data, y_data


def data_pipeline(directory, image_size, image_number):
    X, y = get_data(directory, image_size, image_number)
    X = np.array(X)
    y = np.array(y)
    return X, y

def get_score(X_train, y_train, size):
    #normalization
    X_train = ((X_train / 255.)-.5)*2

    # print("before: ", X_train.shape)
    X_train = X_train.reshape((len(X_train), size*size))
    # print("after: ", X_train.shape)

    #one-hot encoding
    y_train = to_categorical(y_train)

    #from the training set, create a train and test set.
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.22, random_state = 42)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = .25, random_state = 42)

    model = Sequential()
    model.add(Dense(28*14, activation = 'relu'))
    model.add(Dense(14*14, activation = 'relu'))
    model.add(Dense(14*7, activation = 'relu'))
    model.add(Dense(7*7, activation = 'relu'))
    model.add(Dense(27, activation = 'softmax'))

    sgd_opt = SGD(learning_rate=.005, name="SGD")

    model.compile(optimizer=sgd_opt, loss='categorical_crossentropy', metrics = ['accuracy'])

    print("X Train: ", X_train, len(X_train), len(X_train[0]))
    print("y Train: ", y_train, len(y_train), len(y_train[0]))
    print("y val: ", y_val, len(y_val), len(y_val[0]))
    history = model.fit(x=X_train, y=y_train, epochs=30, batch_size = 10, verbose=1, workers=-1, validation_data=(X_val, y_val))
    
    print(history)

    predictions = model.predict(X_test)
    print("Predictions:")
    print(predictions)
    print(" ")
    y_pred = np.rint(predictions)

    acc_sc = accuracy_score(y_test, y_pred)
    f1_sc = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy Score: {acc_sc}. F1 Score: {f1_sc}')

    print(classification_report(y_test, y_pred, zero_division=0))

sizes = [28, 56, 180]
numbers = [0, 1, 3, 5]
# directory_path = r"C:\Users\emwil\OneDrive\Documents\Capstone\test"
# directory_path = r"C:\Users\emwil\OneDrive\Documents\Capstone\filled_in"
# directory_path = r"C:\Users\emwil\OneDrive\Documents\Capstone\filled_in - Copy"
directory_path = r"C:\Users\emwil\OneDrive\Documents\Capstone\hhd_cleaned+merged"
# X_train, y_train = data_pipeline(directory_path, 180, 3)
# print(X_train)
# print("---------------------------------")
# print(y_train)
# print("X shape: ", X_train.shape)
# print("y shape: ", y_train.shape)
# print("Building Model and Getting Score:")
# get_score(X_train, y_train, 180)
for size in sizes:
  for num in numbers:
    print(size, num)
    print(" ")
    X_train, y_train = data_pipeline(directory_path, size, num)
    np.save(f'X_{size}_{num}.npy', X_train)
    np.save(f'y_{size}_{num}.npy', y_train)
#     # print("Now getting scores")
#     print("--------------------------------")
#     # get_score(X_train, y_train, size)

# X_train = np.load('X_28_3.npy')
# y_train = np.load('y_28_3.npy')
# get_score(X_train, y_train, 28)