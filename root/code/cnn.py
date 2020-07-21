import os
import random
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
import cv2
import numpy as np
from tensorflow_core.python.keras.models import load_model


class CNN:
    CATEGORIES = ['circle', 'square', 'star', 'triangle']

    def __init__(self, img_size):
        self.IMG_SIZE = img_size
        self.X = []
        self.y = []
        self.X_train = ''
        self.X_test = ''
        self.y_train = ''
        self.y_test = ''
        self.pickle_out = ''
        self.model = ''
        self.training_data = []

        # evaluation
        # self.model_execution(.2, 64, 32, 3, 2, 1)

    # noinspection PyBroadException
    def create_training_data(self):
        for category in self.CATEGORIES:
            path = os.path.join(r'C:\Users\SHYROKOA\PycharmProjects\MasterThesis\root\image_heap\shapes', category)
            class_num = self.CATEGORIES.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([new_array, class_num])
                except Exception as e:
                    print(f'Exception: {e}')

    def get_training_data_length(self):
        return len(self.training_data)

    def shuffle(self):
        random.shuffle(self.training_data)

    def create_x_y(self):
        for features, label in self.training_data:
            self.X.append(features)
            self.y.append(label)
        self.X = np.array(self.X).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)

    def object_serialization(self):
        pickle_out = open('X.pickle', 'wb')
        pickle.dump(self.X, pickle_out)
        pickle_out.close()

        pickle_out = open('y.pickle', 'wb')
        pickle.dump(self.y, pickle_out)
        pickle_out.close()

    def load_objects(self):
        pickle_in = open('X.pickle', 'rb')
        self.X = pickle.load(pickle_in)

        pickle_in = open('y.pickle', 'rb')
        self.y = pickle.load(pickle_in)

    def split_data(self, test_size):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=test_size,
                                                                                random_state=42)

    def test_split_info(self):
        print(f'X_train: {len(self.X_train)}/n'
              f'X_test: {len(self.X_test)}/n'
              f'y_train: {len(self.y_train)}/n'
              f'y_test: {len(self.y_test)}')

    def one_hot_encode(self):
        # one-hot encode target column
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

    def create_model(self, first_layer, second_layer, k_size, p_size):
        self.model = Sequential()
        self.model.add(Conv2D(first_layer,
                              kernel_size=k_size,
                              activation='relu',
                              input_shape=(self.IMG_SIZE, self.IMG_SIZE, 1)))
        self.model.add(MaxPooling2D(pool_size=(p_size, p_size)))
        self.model.add(Conv2D(second_layer,
                              kernel_size=k_size,
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(p_size, p_size)))
        self.model.add(Flatten())
        self.model.add(Dense(4, activation='softmax'))

    def compile_model(self):
        # compile model using accuracy to measure model performance
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit_model(self, epochs):
        # train the model
        self.model.fit(self.X_train,
                       self.y_train,
                       validation_data=(self.X_test, self.y_test),
                       epochs=epochs)

    def model_execution(self, test_size,first_layer, second_layer, k_size, p_size, epochs):
        self.create_training_data()
        self.shuffle()
        self.create_x_y()
        self.object_serialization()
        self.load_objects()
        self.split_data(test_size)
        self.one_hot_encode()
        self.create_model(first_layer, second_layer, k_size, p_size)
        self.compile_model()
        self.fit_model(epochs)

    def model_save(self):
        self.model.save('my_model')

    def load_model(self):
        self.model = load_model('my_model')


def check(img, size, model):
    dim = (size, size)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # resized = cv2.bitwise_not(resized)
    ret, resized = cv2.threshold(resized, 20, 255, cv2.THRESH_BINARY)
    a = resized
    a = np.expand_dims(a, axis=(0, 3))
    arr = model.model.predict(a)
    arr = arr.tolist()[0]
    print(arr)
    if arr.index(max(arr)) == 0:
        return 'circle'
    elif arr.index(max(arr)) == 1:
        return 'square'
    elif arr.index(max(arr)) == 2:
        return 'star'
    elif arr.index(max(arr)) == 3:
        return 'triangle'
    else:
        return 'Error!'
