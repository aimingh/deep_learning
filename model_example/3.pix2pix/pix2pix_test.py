import tensorflow as tf
from pix2pix_util import pix2pix_GAN, image_flow_from_dir
import os, shutil

checkpoint_dir = './training_checkpoints'
test_dir = "./datasets/cityscapes/val"
result_dir = "./datasets/cityscapes/test_result"

BATCH_SIZE = 64
IMG_WIDTH=256
IMG_HEIGHT=256
OUTPUT_CHANNELS=3

if os.path.exists(result_dir):  # 반복적인 실행을 위해 디렉토리를 삭제합니다.
    shutil.rmtree(result_dir)   # 이 코드는 책에 포함되어 있지 않습니다.
os.mkdir(result_dir)

num_test_img = len(os.listdir(test_dir))
step_for_epoch = num_test_img//BATCH_SIZE if num_test_img%BATCH_SIZE==0 else (num_test_img//BATCH_SIZE + 1)

data = image_flow_from_dir(IMG_WIDTH=IMG_WIDTH,
                            IMG_HEIGHT=IMG_HEIGHT)

model = pix2pix_GAN(OUTPUT_CHANNELS=OUTPUT_CHANNELS, 
                    checkpoint_dir=checkpoint_dir,
                    testmode=True)

test_dataset = tf.data.Dataset.list_files(test_dir + "/*.jpg")
test_dataset = test_dataset.map(data.load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

step = 0
for inp, tar, filename in test_dataset.take(step_for_epoch):
    model.generate_images(model.generator, inp, tar, filename, result_dir)
    print("test step: " + str(step))
