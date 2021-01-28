# import tensorflow as tf
# from tensorflow.keras import models
# from tools.utils import flow_from_dir_concat_images, step_for_epoch, generate_images
# import os, shutil
# import matplotlib.pyplot as plt
# import numpy as np
# from tensorflow.keras.preprocessing import image


# def predict_one_keras():
#     # Model 불러오기
#     model = models.load_model('Unet_small_trt')
#     # model.summary()

#     img = plt.imread("./datasets/cityscapes/val/1.jpg")
#     img_tensor = image.img_to_array(img[:,:256,:])
#     img_tensor = np.expand_dims(img_tensor, axis=0)
#     img_tensor = (img_tensor / 127.5) - 1
#     img_tensor.shape

#     prediction = model.predict(img_tensor)
#     img_seg = prediction[0,:,:,0]
#     plt.imsave("tmp.jpg", img_seg)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
import time
'''
참고
https://eehoeskrap.tistory.com/414
https://colab.research.google.com/github/vinhngx/tensorrt/blob/vinhn-tf20-notebook/tftrt/examples/image-classification/TFv2-TF-TRT-inference-from-Keras-saved-model.ipynb#scrollTo=rf97K_rxvwRm
https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
'''

def predict_tftrt():
    input_saved_model = './Unet_small_trt'

    # img_path = "./datasets/cityscapes/val/1.jpg"
    # img = image.load_img(img_path, target_size=(256, 256, 3))
    img = plt.imread("./datasets/cityscapes/val/1.jpg")
    img_tensor = image.img_to_array(img[:,:256,:])
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = (img_tensor / 127.5) - 1
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)
    # img_tensor = np.float16(img_tensor)
    # img_tensor = tf.constant(img_tensor)

    saved_model_loaded = tf.saved_model.load(input_saved_model, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)

    # result = infer(img_tensor)
    # img_seg = result['conv2d_14'][0,:,:,0]
    # plt.imsave("tmp.jpg", img_seg)

    times = []
    for i in range(100):
        start_time = time.time()
        result = infer(img_tensor)
        delta = (time.time() - start_time)
        print('{} sec:{}'.format(i,delta))
        times.append(delta)

        img_seg = result['conv2d_14'][0,:,:,0]
        img_seg = np.stack((img_seg,)*3, 2)
        result_img = np.hstack((img_tensor[0], img_seg))

    mean_delta = np.array(times).mean()
    print('sec:{}'.format(mean_delta))


if __name__ == "__main__":
    predict_tftrt()