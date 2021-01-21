import tensorflow as tf
from pix2pix_util import pix2pix_GAN, image_flow_from_dir

PATH = "./datasets/cityscapes/"
checkpoint_dir = './training_checkpoints'
log_dir="logs/"
BUFFER_SIZE = 400
BATCH_SIZE = 64
EPOCHS = 150
IMG_WIDTH=256
IMG_HEIGHT=256
OUTPUT_CHANNELS=3
LAMBDA=100

data = image_flow_from_dir(IMG_WIDTH=IMG_WIDTH, IMG_HEIGHT=IMG_HEIGHT)
model = pix2pix_GAN(OUTPUT_CHANNELS=OUTPUT_CHANNELS, 
                    LAMBDA=LAMBDA,
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir)
# data load
train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(data.load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# model graph to png
tf.keras.utils.plot_model(model.generator, show_shapes=True, dpi=64, to_file='pix2pix_gen.png')
tf.keras.utils.plot_model(model.discriminator, show_shapes=True, dpi=64, to_file='pix2pix_disc.png')

# training
model.fit(train_dataset, EPOCHS)