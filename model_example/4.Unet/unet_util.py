import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt

class flow_from_dir_concat_images:
    def __init__(self, IMG_WIDTH=256, IMG_HEIGHT=256):
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT

    def load(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)

        w = tf.shape(image)[1]
        w = w // 2
        l_image = image[:, :w, :]
        r_image = image[:, w:, :]

        r_image = tf.cast(r_image, tf.float32)
        l_image = tf.cast(l_image, tf.float32)
        return l_image, r_image

    def resize(self, l_image, r_image, height, width):
        l_image = tf.image.resize(l_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        r_image = tf.image.resize(r_image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return l_image, r_image

    def random_crop(self, l_image, r_image):
        stacked_image = tf.stack([l_image, r_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3])
        return cropped_image[0], cropped_image[1]

    def normalize(self, l_image, r_image):
        l_image = (l_image / 127.5) - 1
        r_image = (r_image / 127.5) - 1
        return l_image, r_image

    @tf.function()
    def random_jitter(self, l_image, r_image):
        # resize, random crop, random flip 
        l_image, r_image = self.resize(l_image, r_image, 286, 286)
        l_image, r_image = self.random_crop(l_image, r_image)

        if tf.random.uniform(()) > 0.5:
            l_image = tf.image.flip_left_right(l_image)
            r_image = tf.image.flip_left_right(r_image)
        return l_image, r_image

    def load_image_train(self, image_file):
        l_image, r_image = self.load(image_file)
        l_image, r_image = self.random_jitter(l_image, r_image)
        l_image, r_image = self.normalize(l_image, r_image)
        return l_image, r_image

    def load_image_test(self, image_file):
        l_image, r_image = self.load(image_file)
        l_image, r_image = self.resize(l_image, r_image, self.IMG_HEIGHT, self.IMG_WIDTH)
        l_image, r_image = self.normalize(l_image, r_image)
        return l_image, r_image, image_file

def step_for_epoch(num, batch_size):
    return num//batch_size if num%batch_size==0 else (num//batch_size + 1)

def generate_images(model, test_input, tar, filenames, result_dir):
    prediction = model.predict(test_input)

    filenames = np.array(filenames)
    for i in range(len(np.array(filenames))):
        final_img = np.hstack((test_input[i], tar[i]))
        final_img = np.hstack((final_img* 0.5 + 0.5, prediction[i]))
        filename = filenames[i].decode ('utf-8').split('/')[-1][:-4]
        plt.imsave(result_dir + '/' + filename + '.png', final_img )
