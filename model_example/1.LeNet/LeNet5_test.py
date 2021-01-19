# simple classification using LeNet5-like model and mnitst dataset
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# dataset mnist dataset
# train dataset 60000, test dataset 10000
# image size (28,28,1)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)

# Model 불러오기
model = models.load_model('LeNet5_mnist.h5')
model.summary()

# Test 결과
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test loss: {:0.4f}, test accuracy: {:0.2f}%".format(test_loss, 100*test_acc))