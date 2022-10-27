import sys
sys.path.append('/global/cscratch1/sd/jianyao/ForSE/')
from forse.networks.dcgan import *


dcgan = DCGAN(output_directory='/global/cscratch1/sd/jianyao/ForSE/', img_size=(320, 320), slack=True, slack_talk=False)
training_path = '/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/'
training_file = 'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy'
patch_file = training_path+training_file
dcgan.train(epochs=43000, patches_file=patch_file, batch_size=16, save_interval=1000)
# dcgan.train(epochs=101, patches_file=patch_file, batch_size=16, save_interval=1000)
