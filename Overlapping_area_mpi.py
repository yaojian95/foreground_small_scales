### calculating MFs for maps at 3 arcmins

import sys
sys.path.append('/pscratch/sd/j/jianyao/forse_codes/') 
# sys.path.append('/global/cscratch1/sd/jianyao/ForSE/')
from forse.tools.img_tools import *
from forse.tools.mix_tools import *

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


Ls_U = np.load('/pscratch/sd/j/jianyao/forse_output/training_data_3amin.npy')[1]
Thr_U = np.load('/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/GNILC_Thr12_Ulr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1]


Ls_U_rescaled = np.zeros_like(Ls_U)
for i in range(Ls_U.shape[0]):
        
        Ls_U_rescaled[i] = rescale_min_max(Ls_U[i])

Ls_U_rescaled = Ls_U_rescaled.reshape((Ls_U.shape[0], Ls_U.shape[1], Ls_U.shape[1], 1)) # (8526, 320, 320, 1)




dir_models = '/pscratch/sd/j/jianyao/forse_output/U_modify_Adam_lr_5_1e-6/models/'

rhos_t, f_t, u_t, chi_t = [], [], [], []
npatches = 174
for i in range(0,348):

    mT = rescale_min_max(Thr[i], return_min_max=False)
    rhos_T, f_T, u_T, chi_T= get_functionals(mT)
    
    f_t.append(f_T);  u_t.append(u_T); chi_t.append(chi_T)

f_t = np.array(f_t); u_t = np.array(u_t); chi_t = np.array(chi_t)

with open("MFs/U_MFs_lr_5e-6.txt", "a") as o:
    
    for k in range(1000, 100001, 500):

        generator_Q = tf.keras.models.load_model(dir_models+'generat_%s.h5'%k) 
        NNout_Q = generator_Q.predict(Ls_scaled_U)

        rhos_nn, f_nn, u_nn, chi_nn = [], [], [], []  

        for i in range(0,8526):

            mNN = rescale_min_max(NNout_Q[i,:,:,0], return_min_max=False)
            rhos_NN, f_NN, u_NN, chi_NN= get_functionals(mNN)

            f_nn.append(f_NN); u_nn.append(u_NN);chi_nn.append(chi_NN); 

        f_nn = np.array(f_nn); u_nn = np.array(u_nn); chi_nn = np.array(chi_nn); 

        m1_nnq = compute_intersection(rhos_Y, 
                         [np.mean(f_t, axis=0)-np.std(f_t, axis=0), np.mean(f_t, axis=0)+np.std(f_t, axis=0)], 
                         [np.mean(f_nn, axis=0)-np.std(f_nn, axis=0),np.mean(f_nn, axis=0)+np.std(f_nn, axis=0)], 
                         npt=100000)
        m2_nnq = compute_intersection(rhos_Y, 
                             [np.mean(u_t, axis=0)-np.std(u_t, axis=0), np.mean(u_t, axis=0)+np.std(u_t, axis=0)], 
                             [np.mean(u_nn, axis=0)-np.std(u_nn, axis=0),np.mean(u_nn, axis=0)+np.std(u_nn, axis=0)], 
                             npt=100000)
        m3_nnq = compute_intersection(rhos_Y, 
                             [np.mean(chi_t, axis=0)-np.std(chi_t, axis=0), np.mean(chi_t, axis=0)+np.std(chi_t, axis=0)], 
                             [np.mean(chi_nn, axis=0)-np.std(chi_nn, axis=0),np.mean(chi_nn, axis=0)+np.std(chi_nn, axis=0)], 
                             npt=100000)
        
        o.write('%d %.2f %.2f %.2f\n'%(k, m1_nnq, m2_nnq, m3_nnq))
