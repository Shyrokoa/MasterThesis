from root.code.cnn import CNN


cnn = CNN(15)
cnn.model_execution(.2, 64, 32, 3, 2, 30)
cnn.model_save()
