import os, random
import cv2


class CNN:
    CATEGORIES = ['circle', 'square', 'star', 'triangle']
    IMG_SIZE = 25

    def __init__(self):
        self.training_data = []
        self.create_training_data()
        self.shuffle()

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
