from forse_horovod import *

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

import sys
sys.path.append('/pscratch/sd/j/jianyao/forse_codes/') 

from forse.tools.nn_tools import *
# from forse.tools.img_tools import *
# from forse.tools.mix_tools import *

import horovod.tensorflow.keras as hvd
    

dcgan = DCGAN(output_directory='/pscratch/sd/j/jianyao/forse_output/Q_12amin_distributed_sbatch/', img_size=(320, 320), slack=True, slack_talk=False)

training_path = '/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/'
training_file = 'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy'
patch_file = training_path+training_file

# patch_file = '/pscratch/sd/j/jianyao/forse_output/training_data_3amin.npy'
# dcgan.train(opt, gcb, dcb, ccb, epochs=1000, patches_file=patch_file, batch_size=48, save_interval=100)

def train(dcgan, epochs, patches_file, batch_size=32, save_interval=100, seed=4324):
    
    # Horovod: initialize Horovod.
    hvd.init()
    print('size:', hvd.size())
    print('rank:', hvd.local_rank())

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    opt = Adam(0.000005 * hvd.size(), 0.2)
    opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1)

    # callbacks for generator, discriminator, and combined model
    gcb =hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    dcb =hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    ccb =hvd.callbacks.BroadcastGlobalVariablesCallback(0)
    
    generator, discriminator, combined = dcgan.build_gan(opt)

    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    gcb = gcb; dcb = dcb;  ccb = ccb;
    gcb.set_model(generator); dcb.set_model(discriminator); ccb.set_model(combined)
    gcb.on_train_begin(); dcb.on_train_begin(); ccb.on_train_begin()

    X_train, X_test, Y_train, Y_test = load_training_set(patches_file, part_train = 1, part_test = 0, seed=seed)
    print("Training Data Shape: ", X_train.shape)
    half_batch = batch_size // 2

    if dcgan.slack:
        dcgan.client.chat_postMessage(channel = 'ml-training', text = '---------------------------------------------------------')
        dcgan.client.chat_postMessage(channel = 'ml-training', text = 'Training at rank %s start at %.19s'%(hvd.rank(), datetime.now()))
        dcgan.client.chat_postMessage(channel = 'ml-training', text = '---------------------------------------------------------')
        
    # if hvd.rank() == 0:
    #     discriminator.summary()
        
    start = time.time()    
    for epoch in range(epochs):

        if hvd.rank()==0:
            epoch_time = time.time()
            if epoch % 5000 == 0 and epoch > 0:
                message = 'You are at epoch %s ! Time cost is %0.2f mins! ETA: %0.2f hours!'%(epoch, (epoch_time-start)/60, (epochs - epoch)*(epoch_time-start)/60/60/epoch)

                if dcgan.slack:
                    dcgan.client.chat_postMessage(channel = 'ml-training', text = message)
        
        ind_batch = np.random.randint(0, X_train.shape[0], batch_size)
        
        g_loss = combined.train_on_batch(X_train[ind_batch], np.ones((batch_size, 1)))
        target_real = np.ones((half_batch, 1))
        target_fake = np.zeros((half_batch, 1))
        idxX = np.random.randint(0, X_train.shape[0], half_batch)
        idxY = np.random.randint(0, X_train.shape[0], half_batch)
        imgs = Y_train[idxY]
        gen_imgs = generator.predict(X_train[idxX])
        d_loss_real = discriminator.train_on_batch(imgs, target_real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, target_fake)

        if epoch % (save_interval) == 0 and epoch > 0:
            # print('rank:', hvd.rank(), ind_batch)
            if hvd.rank()==0:
                print(epoch)

                save_path = dcgan.output_directory + "/models"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                discriminator.save(save_path + '/discrim_'+str(epoch)+'.h5')
                generator.save(save_path + '/generat_'+str(epoch)+'.h5')
    if hvd.rank()==0:
        discriminator.save(save_path + '/discrim_'+str(epoch)+'.h5')
        generator.save(save_path + '/generat_'+str(epoch)+'.h5')
        
train(dcgan, epochs=1000001, patches_file=patch_file, batch_size=48, save_interval=500)