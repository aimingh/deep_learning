import tensorflow as tf
from tensorflow.keras import models
from tools.utils import flow_from_dir_concat_images, step_for_epoch, generate_images
import os, shutil, time
import numpy as np

IMG_WIDTH = 256
IMG_HEIGHT = 256
BUFFER_SIZE = 400
BATCH_SIZE = 32
EPOCH = 100
PATH = "./datasets/cityscapes/val"
result_dir = "./datasets/cityscapes/test_result_unet"

num_sample = len(os.listdir(PATH))



if __name__ == "__main__":
    if os.path.exists(result_dir): 
        shutil.rmtree(result_dir) 
    os.mkdir(result_dir)

    data = flow_from_dir_concat_images(IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT)
    test_dataset = tf.data.Dataset.list_files(PATH + "/*.jpg")
    test_dataset = test_dataset.map(data.load_image_test)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Model 불러오기
    model = models.load_model('Unet_small.h5')
    model.summary()

    times = []
    for inp, tar, filename in test_dataset.take(step_for_epoch(num_sample, BATCH_SIZE)):
        start_time = time.time()
        generate_images(model, inp, tar, filename, result_dir)
        delta = (time.time() - start_time)
        times.append(delta)

    mean_delta = np.array(times).mean()
    fps = BATCH_SIZE /mean_delta
    print('average(sec):{},fps:{}'.format(mean_delta,fps))

    tf.keras.backend.clear_session()