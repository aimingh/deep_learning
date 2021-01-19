# simple classification using LeNet5-like model and mnitst dataset
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical, plot_model

# dataset mnist dataset
# keras에서 제공해주는 데이터셋을 사용하는 경우
# train dataset 60000, test dataset 10000
# image size (28,28,1)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def preprocessing(images, labels):
    n, w, h = images.shape
    images = images.reshape((n, w, h, 1))
    images = images.astype('float32') / 255
    labels = to_categorical(labels)
    return images, labels

train_images, train_labels = preprocessing(train_images, train_labels)
test_images, test_labels = preprocessing(test_images, test_labels)

# Model 정의
model = models.Sequential()
model.add(layers.Conv2D(filters=6, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(filters=16, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(84, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 정보
model.summary()

# 모델 그래프 이미지로 저장
plot_model(model, show_shapes=True, to_file='LeNet5.png')

# tensorboard 모니터링을 위한 callback 함수 정의
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='log_LeNet',
        histogram_freq=1,
        embeddings_freq=1
    )
]

# Model fitting, training
history = model.fit(train_images, train_labels, 
                    epochs=30, 
                    batch_size=64,
                    validation_split=0.1,
                    callbacks=callbacks)

# 학습된 model 저장
model.save('LeNet5_mnist.h5')

# Test 결과
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test loss: {:0.4f}, test accuracy: {:0.2f}%".format(test_loss, 100*test_acc))