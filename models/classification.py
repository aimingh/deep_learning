# deep learning models
from tensorflow.keras import layers, models

# LeNet5 model
def lenet5(class_num=10, input_size = (28, 28, 1)):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=6, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(class_num, activation='softmax'))
    return model

# Alexnet model with batch norm
def alexnet(class_num=1000, input_size = (224,224,3)):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=96, kernel_size=(11,11), strides=4, padding='same', activation='relu', input_shape=input_size)
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=256, kernel_size=(5,5), strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(class_num, activation='sigmoid'))
    return model

def resnet50(class_num=1000, input_size = (224,224,3)):
    model = models.Sequential()
    return model

def mobilenet(class_num=1000, input_size = (224,224,3)):
    model = models.Sequential()
    return model