import os
from PIL import Image

train_directory = '.\hhd_dataset_cleaned\TRAIN'
test_directory = '.\hhd_dataset_cleaned\TEST'
 
def get_data(directory):
    X_data = []
    y_data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isdir(f):
            temp_X_data, temp_y_data = get_data(f)
            for i in temp_X_data:
                X_data.append(i)
            for i in temp_y_data:
                y_data.append(i)
        elif os.path.isfile(f):
            X_data.append(Image.open(f))
            strings = f.split("\\")
            file_ending = strings[len(strings)-1]
            letter_number = file_ending.split('_')[0]
            y_data.append(int(letter_number))
    return X_data, y_data

X_train, y_train = get_data(train_directory)

print(len(X_train))
print(len(y_train))

X_test, y_test = get_data(test_directory)

print(len(X_test))
print(len(y_test))
