
from root.code.image import Image
from root.code.cnn import CNN

cnn = CNN(15)
cnn.load_model()

img = Image(r'C:\Users\SHYROKOA\PycharmProjects\MasterThesis\root\image_heap\sym.jpg', 15, cnn)
