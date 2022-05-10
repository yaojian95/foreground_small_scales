import sys
sys.path.append('/pscratch/sd/j/jianyao/forse_codes/') 
from forse.tools.nn_tools import *

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import UpSampling2D, Conv2D, Activation, BatchNormalization
from tensorflow.keras.layers import Reshape, Dense, Input
from tensorflow.keras.layers import LeakyReLU, Dropout, Flatten, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to avoid printing lots of info of GPU

import re
from datetime import datetime
import time

import horovod
import horovod.tensorflow as hvd


dirs = '/pscratch/sd/j/jianyao/forse_output/3_arcmin_8000_models_MY_lr_5e-5_partT_1/'
img_shape = (320, 320, 1); kernel_size = 5; cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True); channels = 1;
def build_generator():
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=kernel_size, padding="same", input_shape=img_shape)) # 64x64x64
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Conv2D(128, kernel_size=kernel_size, padding="same", strides=2)) #32x32x128
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Conv2D(256, kernel_size=kernel_size, padding="same", strides=2)) #16x16x256
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=kernel_size, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=kernel_size, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Conv2D(channels, kernel_size=kernel_size, padding="same"))
    model.add(Activation("tanh"))
    # img_in = Input(shape=img_shape)
    # img_out = model(img_in)
    # return Model(img_in, img_out)
    return model

def build_discriminator():

    model = Sequential()
    model.add(Conv2D(64, kernel_size=kernel_size, strides=1, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Conv2D(128, kernel_size=kernel_size, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Conv2D(256, kernel_size=kernel_size, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))
    # img = Input(shape=img_shape)
    # validity = model(img)
    # return Model(img, validity)
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(0.00005* hvd.size(), 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(0.00005* hvd.size(), 0.5)

checkpoint_dir = dirs + 'training_checkpoints'
if hvd.rank() == 0:  
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

### recover from the latest training epoch 
latest = tf.train.latest_checkpoint(checkpoint_dir);
if latest is not None:
    epoch_latest = np.int(re.findall(r'\d+', latest)[-1])*500
    print('restore from checkpoint:%s'% latest)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

@tf.function
def train_step(noise, images, first_epoch):
    # noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gen_tape = hvd.DistributedGradientTape(gen_tape)
    disc_tape = hvd.DistributedGradientTape(disc_tape)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    if first_epoch:
            hvd.broadcast_variables(generator.variables, root_rank=0)
            hvd.broadcast_variables(generator_optimizer.variables(), root_rank=0)
            hvd.broadcast_variables(discriminator.variables, root_rank=0)
            hvd.broadcast_variables(discriminator_optimizer.variables(), root_rank=0)
            
def train(output_directory, epochs, patches_file, batch_size=32, save_interval=100, seed=4324):
    start = time.time()
    
    X_train, X_test, Y_train, Y_test = load_training_set(patches_file, part_train = 1, part_test = 0, seed=seed); # X-input large scales; Y-real small scales
    print("Training Data Shape: ", X_train.shape)
    half_batch = batch_size // 2
    accs = []
    
    if latest:
        epochs_range = range(epoch_latest, epochs)
        np.random.randint(0, X_train.shape[0], 48*(epoch_latest)) #To let the random start from lastest state, not the 0-state
    else:
        epochs_range = range(epochs)
    
    for epoch in epochs_range:
        now = time.time()
        
        ind_batch = np.random.randint(0, X_train.shape[0], batch_size)

        idxX = np.random.randint(0, X_train.shape[0], half_batch)
        idxY = np.random.randint(0, X_train.shape[0], half_batch)

        train_step(X_train[ind_batch], Y_train[idxY], epoch == 0)
        
        if hvd.rank() == 0:
            if epoch < 10:
                print(ind_batch)
            if (epoch + 1) % save_interval == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
                message = 'You are at epoch %s ! Time cost is %0.2f mins! ETA: %0.2f hours!'%(epoch, (now-start)/60, (epochs - epoch)*(now-start)/60/60/epoch)
                print(message)
            
# training_path = '/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/'
# training_file = 'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy'
# patch_file = training_path+training_file

patch_file = '/pscratch/sd/j/jianyao/forse_output/training_data_3amin.npy'

train(output_directory= dirs, epochs=200001, patches_file=patch_file, batch_size=48, save_interval=500)
