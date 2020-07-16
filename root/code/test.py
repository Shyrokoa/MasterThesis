from root.code.cnn import CNN

cnn = CNN()
cnn.shuffle()
for sample in cnn.training_data:
    print(sample[1])
