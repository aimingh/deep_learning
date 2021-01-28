import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tools.segmentation import unet
from unet_util import flow_from_dir_concat_images, step_for_epoch
import os

IMG_WIDTH = 256
IMG_HEIGHT = 256
BUFFER_SIZE = 400
BATCH_SIZE = 32
EPOCH = 100
PATH = "./datasets/cityscapes/train"

if __name__ == "__main__":
    data = flow_from_dir_concat_images(IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT)
    train_dataset = tf.data.Dataset.list_files(PATH+'/*.jpg')
    train_dataset = train_dataset.map(data.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    model = unet((256,256,3), 
                n_filters = 32, 
                kernel_size=3, 
                dropout_rate=0.1,
                output_channerl = 3)

    model.compile(optimizer='adam',
                loss='mse',
                metrics=['mse'])

    model.summary()
    plot_model(model, show_shapes=True, to_file='unet.png')

    callbacks = [keras.callbacks.TensorBoard(
                    log_dir='Unet',
                    histogram_freq=1,
                    write_graph=True,
                    update_freq=1,)]

    history = model.fit(train_dataset,
                        epochs=EPOCH,
                        callbacks=callbacks)

    model.save('Unet.h5')
