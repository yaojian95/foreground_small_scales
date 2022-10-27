import sys
sys.path.append('/pscratch/sd/j/jianyao/forse_codes/') 
# sys.path.append('/global/cscratch1/sd/jianyao/ForSE/')
from forse.tools.img_tools import *
from forse.tools.mix_tools import *

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prepare_small_scales(Ls_Q, Ls_U, Ss_gauss_Q, Ss_gauss_U):
    
    npatches = len(Ls_Q)
    ls_Q = np.zeros_like(Ls_Q); ls_U = np.zeros_like(Ls_Q); ss_gauss_Q = np.zeros_like(Ls_Q); ss_gauss_U = np.zeros_like(Ls_Q)
    ss_gauss_Q_ratio, ss_gauss_U_ratio= Ss_gauss_Q/Ls_Q, Ss_gauss_U/Ls_U # Remove large scales (80') from the gaussian maps, according the definition of ss in the paper
    for i in range(npatches):
        
        ls_Q[i], ls_U[i] = rescale_min_max(Ls_Q[i]), rescale_min_max(Ls_U[i])
        ss_gauss_Q[i], ss_gauss_U[i] = rescale_min_max(ss_gauss_Q_ratio[i]), rescale_min_max(ss_gauss_U_ratio[i])
        
    return ls_Q.reshape((npatches, 320, 320, 1)), ls_U.reshape((npatches, 320, 320, 1)), ss_gauss_Q.reshape((npatches, 320, 320, 1)), ss_gauss_U.reshape((npatches, 320, 320, 1)),ss_gauss_Q_ratio,ss_gauss_U_ratio

dir_data = '/global/cfs/cdirs/sobs/www/users/ForSE/NN_datautils/datasets/'

Thr, Ls_Q = np.load(dir_data+'GNILC_Thr12_Qlr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[:, 0:174]
Ls_U = np.load(dir_data+'GNILC_Thr12_Ulr80_20x20deg_Npix320_full_sky_adaptive.npy', allow_pickle=True)[1, 0:174]

Ss_gaussQ = np.load(dir_data+'GNILC_gaussian_ss_Q_20x20deg_Npix320_full_sky_adaptive.npy',  allow_pickle=True)
Ss_gaussU = np.load(dir_data+'GNILC_gaussian_ss_U_20x20deg_Npix320_full_sky_adaptive.npy',  allow_pickle=True)

Ls_scaled_Q, Ls_scaled_U, Ss_ratio_scaled_Q, Ss_ratio_scaled_U, Ss_ratio_Q, Ss_ratio_U = prepare_small_scales(Ls_Q, Ls_U, Ss_gaussQ, Ss_gaussU)

dir_models = '/pscratch/sd/j/jianyao/forse_output/U_modify_Adam_lr_5_1e-6/models/'
# dir_models = '/pscratch/sd/j/jianyao/ForSE/forse/scripts/models/'

rhos_gss, f_gss, u_gss, chi_gss = [], [], [], []
rhos_t, f_t, u_t, chi_t = [], [], [], []
npatches = 174
for i in range(0,npatches):
    rhos_Y, f_Y, u_Y, chi_Y = get_functionals(Ss_ratio_scaled_Q[i,:,:,0])

    mT = rescale_min_max(Thr[i], return_min_max=False)
    rhos_T, f_T, u_T, chi_T= get_functionals(mT)
    
    f_t.append(f_T);  u_t.append(u_T); chi_t.append(chi_T)
    f_gss.append(f_Y); u_gss.append(u_Y); chi_gss.append(chi_Y); 

f_t = np.array(f_t); u_t = np.array(u_t); chi_t = np.array(chi_t)
f_gss = np.array(f_gss); u_gss = np.array(u_gss); chi_gss = np.array(chi_gss); 

with open("MFs/U_MFs_lr_5e-6.txt", "a") as o:
    
    for k in range(1000, 100001, 500):

        generator_Q = tf.keras.models.load_model(dir_models+'generat_%s.h5'%k) 
        NNout_Q = generator_Q.predict(Ls_scaled_U)

        rhos_nn, f_nn, u_nn, chi_nn = [], [], [], []  

        for i in range(0,npatches):

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