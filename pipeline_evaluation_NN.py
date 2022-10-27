import numpy as np
import pymaster as nmt

import sys
# # sys.path.append('/global/cscratch1/sd/jianyao/ForSE/')
sys.path.append('/pscratch/sd/j/jianyao/forse_codes/') 
from forse.tools.nn_tools import *
from forse.tools.img_tools import *
from forse.tools.mix_tools import *

import matplotlib.pyplot as plt
## load data
## NN small scales, Gaussian small scales(to normalize the NN ss), Larges scales

# maps_out_3Q = np.load('/pscratch/sd/j/jianyao/forse_output/NN_small_scales/NN_Qout_3amin_Nico_20amin_model_from_5_6.npy') 
# maps_out_3U = np.load('/pscratch/sd/j/jianyao/forse_output/NN_small_scales/NN_Uout_3amin_Nico_20amin_model_from_5_6.npy')

# print('out_3Q', maps_out_3Q.shape)
# maps_out_3Q = maps_out_3Q.reshape(174,49,320,320)
# maps_out_3U = maps_out_3U.reshape(174,49,320,320)
# print('out_3Q', maps_out_3Q.shape)

# gauss_ss_ps = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_3_over_20_power_spectra.npy') #[2, 174, 49, 1, 25] Q, U
# gauss_ss_mean_std = np.load('/pscratch/sd/j/jianyao/forse_output/gauss_small_scales_3_over_20_mean_and_std.npy') #[4, 174, 49] Q_mean, Q_std, U_mean, U_std

# Nico_20amin_Q = np.load('/pscratch/sd/j/jianyao/forse_output/Nico_Q_20amin.npy')
# Nico_20amin_U = np.load('/pscratch/sd/j/jianyao/forse_output/Nico_U_20amin.npy')
# print('20Q', Nico_20amin_Q.shape)
# Nico_20amin_Q  = Nico_20amin_Q.reshape(174,49,320,320)
# Nico_20amin_U  = Nico_20amin_U.reshape(174,49,320,320)
# print('20Q', Nico_20amin_Q.shape)

### ----------- renormalization to physical units ----------------

def first_normalization(maps_out_3Q, maps_out_3U, gauss_ss_mean_std):
    '''
    Normalize the small scales w.r.t. Gaussian case in the map level;
    -----------------------------------------------------------------
    maps_out_3Q/U: small scales generated from the Neural Network. With shape: (174, 49, 320, 320);
    gauss_ss_mean_std: mean and std for each patch of small scales of Gaussian realization, defined by the ratio: 
    Gaussian_maps_3amin/Gaussian_maps_20amin; 4 in 1: Q_mean, Q_std, U_mean, U_std. With shape (4, 174, 49).
    
    '''
    
    for i in range(174):
        for j in range(49):
            maps_out_3Q[i, j] = (maps_out_3Q[i, j] - np.mean(maps_out_3Q[i, j]))/np.std(maps_out_3Q[i, j])*gauss_ss_mean_std[1][i, j] + gauss_ss_mean_std[0][i, j]
            maps_out_3U[i, j] = (maps_out_3U[i, j] - np.mean(maps_out_3U[i, j]))/np.std(maps_out_3U[i, j])*gauss_ss_mean_std[3][i, j] + gauss_ss_mean_std[2][i, j]
   
    return maps_out_3Q, maps_out_3U

def second_normalization(maps_out_3Q, maps_out_3U, gauss_ss_ps, gauss_ss_mean_std, Nico_20amin_Q, Nico_20amin_U, lmin = 40*14, lmax = 3500):
    '''
    Normalize the small scales w.r.t. Gaussian case in the power spectra level, after the first normalization.
    ----------------------------------------------------------------------------------------------------------
    maps_out_3Q/U: small scales after the first normalization; With shape: (174, 49, 320, 320);
    gauss_ss_ps: power spectra for each patch of small scales of Gaussian realization; 2 in 1: cl_QQ and cl_UU; with shape: (2, 174, 49, 1, 25).
    
    '''

    Lx = np.radians(20.)
    Ly = np.radians(20.)
    Nx = 320
    Ny = 320

    mask = np.load('mask_320*320.npy')

    l0_bins = np.arange(20, lmax, 40)
    lf_bins = np.arange(20, lmax, 40)+39
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    ells_uncoupled = b.get_effective_ells()

    f_SSQ = nmt.NmtFieldFlat(Lx, Ly, mask, [np.zeros((320, 320))])
    w00 = nmt.NmtWorkspaceFlat()
    w00.compute_coupling_matrix(f_SSQ, f_SSQ, b)

    NNmapQ_corr = np.ones((174, 49, 320, 320))
    NNmapU_corr = np.ones((174, 49, 320, 320))

    for i in range(0, 174):
        for j in range(49):
           
            f_NNQ = nmt.NmtFieldFlat(Lx, Ly, mask, [maps_out_3Q[i, j]])
            cl_NN_coupledQ = nmt.compute_coupled_cell_flat(f_NNQ, f_NNQ, b)
            cl_NN_uncoupledQ = w00.decouple_cell(cl_NN_coupledQ)
            f_NNU = nmt.NmtFieldFlat(Lx, Ly, mask, [maps_out_3U[i, j]])
            cl_NN_coupledU = nmt.compute_coupled_cell_flat(f_NNU, f_NNU, b)
            cl_NN_uncoupledU = w00.decouple_cell(cl_NN_coupledU)
            
            ell_s = int(lmin/40)
            NNmapQ_corr[i,j]=((maps_out_3Q[i,j]-np.mean(maps_out_3Q[i,j]))/np.sqrt(np.mean(cl_NN_uncoupledQ[0][ell_s:]/gauss_ss_ps[0,i,j][0][ell_s:]))+ gauss_ss_mean_std[0][i, j])*Nico_20amin_Q[i, j] 
            NNmapU_corr[i,j]=((maps_out_3U[i,j]-np.mean(maps_out_3U[i,j]))/np.sqrt(np.mean(cl_NN_uncoupledU[0][ell_s:]/gauss_ss_ps[1,i,j][0][ell_s:]))+ gauss_ss_mean_std[2][i, j])*Nico_20amin_U[i, j]
            
    return NNmapQ_corr, NNmapU_corr

## ---------------- map visualization -----------------------

def plot_maps(Nico_20amin, maps_out_3, NNmap_corr,m = 36, n = 4):
    
    '''
    map visualization; maps at 20 amin; output from NN; renormalize the NN output and combine with the large scales
    m: sky_position. 0-174
    n: patch_position in the 7*7 square
    '''
    fig, axes = plt.subplots(5, 3, figsize = (20, 28))
    
    for l in range(5):
        axes[l][0].imshow(Nico_20amin[m+l, n])
        axes[l][1].imshow(maps_out_3[m+l, n])
        axes[l][2].imshow(NNmap_corr[m+l, n])
        
## --------------- MFs -------------------------------

def get_one_MF(Thr, NNout, patch_N):
    '''
    Defined for output at 3amin, [174,49, 320, 320]
    calculate MFs of generated NN small scales and overlapping with Intensity small scales 
    '''
    rhos_t, f_t, u_t, chi_t = [], [], [], []
    npatches = 348
    for i in range(0,npatches):

        mT = rescale_min_max(Thr[i], return_min_max=False)
        rhos_T, f_T, u_T, chi_T= get_functionals(mT)

        f_t.append(f_T);  u_t.append(u_T); chi_t.append(chi_T)

    f_t = np.array(f_t); u_t = np.array(u_t); chi_t = np.array(chi_t)

    rhos_nn, f_nn, u_nn, chi_nn = [], [], [], []  
    
    if patch_N == 0:  ## use two sets of patches (each has 174 patches) to compare with Intensity 348 maps, except for patch_N = 0, which only has 1 set
        i_s = 0; i_e = 1
    else:
        i_s = patch_N*2 - 1; i_e = (patch_N+1)*2 - 1
        
    for i in range(0,174):
        for j in range(i_s, i_e):
            mNN = rescale_min_max(NNout[i,j,:,:], return_min_max=False)
            rhos_NN, f_NN, u_NN, chi_NN= get_functionals(mNN)
            f_nn.append(f_NN); u_nn.append(u_NN);chi_nn.append(chi_NN); 

    f_nn = np.array(f_nn); u_nn = np.array(u_nn); chi_nn = np.array(chi_nn); 

    m1_nnq = compute_intersection(rhos_T, 
                     [np.mean(f_t, axis=0)-np.std(f_t, axis=0), np.mean(f_t, axis=0)+np.std(f_t, axis=0)], 
                     [np.mean(f_nn, axis=0)-np.std(f_nn, axis=0),np.mean(f_nn, axis=0)+np.std(f_nn, axis=0)], npt=100000)
    m2_nnq = compute_intersection(rhos_T, 
                         [np.mean(u_t, axis=0)-np.std(u_t, axis=0), np.mean(u_t, axis=0)+np.std(u_t, axis=0)], 
                         [np.mean(u_nn, axis=0)-np.std(u_nn, axis=0),np.mean(u_nn, axis=0)+np.std(u_nn, axis=0)], npt=100000)
    m3_nnq = compute_intersection(rhos_T, 
                         [np.mean(chi_t, axis=0)-np.std(chi_t, axis=0), np.mean(chi_t, axis=0)+np.std(chi_t, axis=0)], 
                         [np.mean(chi_nn, axis=0)-np.std(chi_nn, axis=0),np.mean(chi_nn, axis=0)+np.std(chi_nn, axis=0)], npt=100000)
    
    return m1_nnq, m2_nnq, m3_nnq, rhos_T, f_t, u_t, chi_t, f_nn, u_nn, chi_nn

def plot_MF(results, S, savedir = False):
    rhos_Y, f_t, u_t, chi_t, f_nn, u_nn, chi_nn = results[3:10] 
    fig, axes = plt.subplots(1,3, figsize=(24, 4))

    for i in range(3):
        f_nn = results[7+i]; f_t = results[4+i]
        axes[i].fill_between(rhos_Y, 
                             np.mean(f_nn, axis=0)-np.std(f_nn, axis=0), 
                             np.mean(f_nn, axis=0)+np.std(f_nn, axis=0), 
                             lw=1, label=r'$m_{ss}^{NN, %s}$'%S, alpha=0.5, color='#F87217')
        axes[i].plot(rhos_Y, np.mean(f_nn, axis=0), lw=3, ls='--', color='#D04A00')
        axes[i].fill_between(rhos_Y, 
                             np.mean(f_t, axis=0)-np.std(f_t, axis=0), 
                             np.mean(f_t, axis=0)+np.std(f_t, axis=0), 
                             lw=2, label = r'$m_{ss}^{real, I}$', edgecolor='black', facecolor='None')
        axes[i].plot(rhos_Y, np.mean(f_t, axis=0), lw=2, ls='--', color='black')
        
        axes[i].set_xlabel(r'$\rho$', fontsize=20)
        axes[i].set_ylabel(r'$\mathcal{V}_{%s}(\rho$) %s'%(i, S), fontsize=20)
        axes[i].set_title('%.2f'%results[i], fontsize = 20)
        if i == 0:
            axes[i].legend(fontsize = 25)
    if savedir:
        plt.savefig(savedir, format = 'pdf')
            
## ------------------- power spectra --------------------

def cl_flat(Lx, Ly, Nx, Ny, mpq, mpu, lmax, mask, beam_amin = None, w22_file = "w22_flat_320_320.fits"):
    '''
    not correcting for the beam: takes about 18mins
    '''
    if mask:
        mask = np.load(mask)
    else:
        mask = np.ones_like(mpq).flatten()
        xarr = np.ones(Ny)[:, None] * np.arange(Nx)[None, :] * Lx/Nx
        yarr = np.ones(Nx)[None, :] * np.arange(Ny)[:, None] * Ly/Ny
        mask[np.where(xarr.flatten() < Lx / 100.)] = 0
        mask[np.where(xarr.flatten() > 99 * Lx / 100.)] = 0
        mask[np.where(yarr.flatten() < Ly / 100.)] = 0
        mask[np.where(yarr.flatten() > 99 * Ly / 100.)] = 0
        mask = mask.reshape([Ny, Nx])
        mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=2., apotype="C1")
        np.save('mask_320*320.npy', mask)

    # define bins for a binned power spectrum
    l0_bins = np.arange(20, lmax, 40)
    lf_bins = np.arange(20, lmax, 40)+39
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    if beam_amin:
        ell_int = b.get_effective_ells().astype('int')
        beams = hp.gauss_beam(beam_amin/60/180*np.pi, lmax = lmax)
        
        f2 = nmt.NmtFieldFlat(Lx, Ly, mask, [mpq, mpu], purify_b=True, beam = [ell_int, beams[ell_int]])
    else:
        f2 = nmt.NmtFieldFlat(Lx, Ly, mask, [mpq, mpu], purify_b=True)

    # compute matrix for power spectrum   
    w22 = nmt.NmtWorkspaceFlat()
    
    try:
        w22.read_from(w22_file)
        print('weights loaded from %s' % w22_file)
    except:
        w22.compute_coupling_matrix(f2, f2, b)
        w22.write_to(w22_file)
        print('weights writing to disk')
    
    cl22_coupled = nmt.compute_coupled_cell_flat(f2, f2, b)
    cl22_uncoupled = w22.decouple_cell(cl22_coupled)
    
    return b.get_effective_ells(), cl22_uncoupled