# AlexNet like model with dogs vs cats dataset
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_dataset = './datasets/dogs_vs_cats_split'
batch_size = 64
num_sample = [15000, 5000, 5000]

# data preprocessing
# 모든 이미지를 1/255로 스케일을 조정합니다
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        path_dataset + '/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        path_dataset + '/validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

test_generator = test_datagen.flow_from_directory(
        path_dataset + '/test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary')

# Model 정의
model = models.Sequential()
model.add(layers.Conv2D(filters=96, kernel_size=(11,11), strides=4, padding='same', activation='relu', input_shape=(224, 224, 3)))
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
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 정보
model.summary()

# 모델 그래프 이미지로 저장
plot_model(model, show_shapes=True, to_file='AlexNet.png')

# tensorboard 모니터링을 위한 callback 함수 정의
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='AlexNet_log',
        histogram_freq=1,
        write_graph=True,
        update_freq=1,
    )
]

# Model fitting, training
history = model.fit_generator(
      train_generator,
      steps_per_epoch=num_sample[0]//batch_size,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=num_sample[1]//batch_size,
      callbacks=callbacks)

# 학습된 model 저장
model.save('AlexNet.h5')

# Test 결과
test_loss, test_acc = model.evaluate_generator(
        generator=test_generator,
        steps=num_sample[2]//batch_size)

print("test loss: {:0.4f}, test accuracy: {:0.2f}%".format(test_loss, 100*test_acc))
