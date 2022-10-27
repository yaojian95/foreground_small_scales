import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt


### load output (small scales only at 3amin) from the neural network 
# :shape: (174*49, 320, 320)

class post_training(object):
    
    def __init__(self, NNout_Q, NNout_U, training_files_Q, training_file_U, MF = True):
        '''
        training_files_Q/U; input training files for the NN, shape:(2, 348, 320, 320)
        '''
        
        self.NNout_Q = NNout_Q;
        self.NNout_U = NNout_U;
        
        self.thr = training_files_Q[0, 0:174]; # intensity small scales at 12amin
        self.Ls_Q = training_files_Q[1, 0:174]*1e6 # Large scales to uk_RJ
        self.Ls_U = training_files_U[1, 0:174]*1e6
        
        if MF:
            self.MF_I = self.get_one_MF(self.thr, npatches = 348, patch_N = False)

    def first_normalization(self, gauss_ss_mean_std):
        '''
        Normalize the small scales w.r.t. Gaussian case in the map level;
        
        Parameters
        -----------------------------------------------------------------
        maps_out_12Q/U: small scales generated from the Neural Network. With shape: (174, 320, 320);
        gauss_ss_mean_std: mean and std for each patch of small scales of Gaussian realization, defined by the ratio: 
        Gaussian_maps_12amin/Gaussian_maps_80amin; 4 in 1: Q_mean, Q_std, U_mean, U_std. With shape (4, 174).
        
        Returns
        -------
        normalized maps.

        '''
        
        NNout_normed_Q, NNout_normed_U = self.NNout_Q, self.NNout_U
        for i in range(174):
            NNout_normed_Q[i] = NNout_normed_Q[i]/np.std(NNout_normed_Q[i])*gauss_ss_mean_std[1][i]
            NNout_normed_Q[i] = NNout_normed_Q[i]-np.mean(NNout_normed_Q[i])+gauss_ss_mean_std[0][i]
            NNout_normed_U[i] = NNout_normed_U[i]/np.std(NNout_normed_U[i])*gauss_ss_mean_std[3][i]
            NNout_normed_U[i] = NNout_normed_U[i]-np.mean(NNout_normed_U[i])+gauss_ss_mean_std[2][i]
    
        return NNout_normed_Q, NNout_normed_U

    def normalization(self, gauss_ss_ps, gauss_ss_mean_std, Ls_Q, Ls_U, lmin = 40*14, lmax = 3500):
        '''
        Normalize the small scales w.r.t. Gaussian case in the power spectra level and multiply with the large scales to get a full-resolution maps, after the first normalization.
        
        Parameters
        ----------
        maps_out_12Q/U: small scales after the first normalization; With shape: (174, 320, 320);
        gauss_ss_ps: power spectra for each patch of small scales of Gaussian realization; 2 in 1: cl_QQ and cl_UU; with shape: (2, 174, 1, 25).
        Ls_Q/U: large scales, same as the input for the training; with shape (348,320,320).
        
        Returns
        -------
        
        patches of full resolution maps with physical units.
        '''
        maps_out_3Q, maps_out_3U = self.first_normalization(gauss_ss_mean_std)
        
        Lx = np.radians(20.); Ly = np.radians(20.)
        Nx = 320; Ny = 320

        mask = np.load('mask_320*320.npy')

        l0_bins = np.arange(20, lmax, 40)
        lf_bins = np.arange(20, lmax, 40)+39
        b = nmt.NmtBinFlat(l0_bins, lf_bins)
        ells_uncoupled = b.get_effective_ells()

        f_SSQ = nmt.NmtFieldFlat(Lx, Ly, mask, [np.zeros((320, 320))])
        w00 = nmt.NmtWorkspaceFlat()
        w00.compute_coupling_matrix(f_SSQ, f_SSQ, b)
        
        NNmapQ_corr = np.ones((174, 320, 320))
        NNmapU_corr = np.ones((174, 320, 320))

        for i in range(0, 174):
            
            f_NNQ = nmt.NmtFieldFlat(Lx, Ly, mask, [maps_out_3Q[i]])
            cl_NN_coupledQ = nmt.compute_coupled_cell_flat(f_NNQ, f_NNQ, b)
            cl_NN_uncoupledQ = w00.decouple_cell(cl_NN_coupledQ)
            f_NNU = nmt.NmtFieldFlat(Lx, Ly, mask, [maps_out_3U[i]])
            cl_NN_coupledU = nmt.compute_coupled_cell_flat(f_NNU, f_NNU, b)
            cl_NN_uncoupledU = w00.decouple_cell(cl_NN_coupledU)

            newQ = maps_out_3Q[N]/np.sqrt(np.mean(cl_NN_uncoupledQ[0][4:]/gauss_ss_ps[0][0][4:]))
            newU = maps_out_3U[N]/np.sqrt(np.mean(cl_NN_uncoupledU[0][4:]/gauss_ss_ps[1][0][4:]))
            newQ = ((newQ)-np.mean(newQ)+gauss_ss_mean_std[0][i])*self.Ls_Q[i]
            newU = ((newU)-np.mean(newU)+gauss_ss_mean_std[1][i])*self.Ls_U[i]
            NNmapQ_corr[i] = newQ
            NNmapU_corr[i] = newU
    
        self.NNmapQ_corr, self.NNmapU_corr = NNmapQ_corr, NNmapU_corr
        return NNmapQ_corr, NNmapU_corr
    
    
    def get_one_MF(self, input_maps, npatches = 174, patch_N = False):
        '''
        Defined for output at 12amin, [174, 320, 320] or for ordinary maps with shape [174, 320, 320]
        for nn output at 12amin, npatches = 174; for intensity small scales, npatch = 174;
        
        Returns
        -------
        rhos: threshold values, normally [-1, 1]
        f, u, chi : three kinds of MFs for each patch
        
        '''
        rhos, f, u, chi = [], [], [], []
        maps_MF = input_maps # for intensity small scales with shape (174, 320, 320)
        
        for i in range(0,npatches):
    
            mT = rescale_min_max(maps_MF[i], return_min_max=False)
            rhos, f, u, chi= get_functionals(mT)

            f.append(f);  u.append(u); chi.append(chi)

        f = np.array(f); u = np.array(u); chi = np.array(chi)
        
        return rhos, f, u, chi
    
    def plot_MF(self, savedir = False):
        
        rhos_Y, f_t, u_t, chi_t = self.MF_I
        rhos_Y, f_nn_Q, u_nn_Q, chi_nn_Q = self.get_one_MF(self.NNmapQ_corr) 
        rhos_Y, f_nn_U, u_nn_U, chi_nn_U = self.get_one_MF(self.NNmapU_corr)       
        
        f_nn_all = [[f_nn_Q, u_nn_Q, chi_nn_Q],[f_nn_Q, u_nn_Q, chi_nn_Q]]
        f_i = [f_t, u_t, chi_t]
        fig, axes = plt.subplots(2,3, figsize=(24, 8))
        S = ['Q', 'U']
        for i in range(3):
            for j in range(2):
                f_nn = f_nn_all[j, i]; f_t = f_i[i];
                
                axes[i, j].fill_between(rhos_Y, 
                                     np.mean(f_nn, axis=0)-np.std(f_nn, axis=0), 
                                     np.mean(f_nn, axis=0)+np.std(f_nn, axis=0), 
                                     lw=1, label=r'$m_{ss}^{NN, %s}$'%S[j], alpha=0.5, color='#F87217')
                axes[i, j].plot(rhos_Y, np.mean(f_nn, axis=0), lw=3, ls='--', color='#D04A00')
                axes[i, j].fill_between(rhos_Y, 
                                     np.mean(f_t, axis=0)-np.std(f_t, axis=0), 
                                     np.mean(f_t, axis=0)+np.std(f_t, axis=0), 
                                     lw=2, label = r'$m_{ss}^{real, I}$', edgecolor='black', facecolor='None')
                axes[i, j].plot(rhos_Y, np.mean(f_t, axis=0), lw=2, ls='--', color='black')

                axes[i, j].set_xlabel(r'$\rho$', fontsize=20)
                axes[i, j].set_ylabel(r'$\mathcal{V}_{%s}(\rho$) %s'%(i, S[j]), fontsize=20)
                axes[i, j].set_title('%.2f'%results[i], fontsize = 20)
                if i == 0:
                    axes[i, j].legend(fontsize = 25)
        if savedir:
            plt.savefig(savedir, format = 'pdf')

    
    def visualization(self, maps_out_12, n):
    
        '''
        map visualization; maps at 20 amin; output from NN; renormalize the NN output and combine with the large scales
        m: sky_position. 0-174
        n: patch_position in the 7*7 square
        '''
        
        '''  needs to be fixed  '''
        fig, axes = plt.subplots(5, 3, figsize = (20, 28))

        for l in range(5):
            axes[l][0].imshow(self.Ls[n+l])
            axes[l][1].imshow(maps_out_12[n+l])
            axes[l][2].imshow(NNmap_corr[n+l])
        
        pass
    
    
    def reproject_to_fullsky(self, ):
        
        salloc --nodes 4 --qos interactive --time 00:30:00 --constraint cpu --account=mp107
        module load tensorflow/2.6.0
        srun -n 16 python reproject2fullsky_mpi.py --pixelsize 3.75 --npix 320 --overlap 2   --verbose  --flat-projection /pscratch/sd/j/jianyao/forse_processed_data/NN_out_Q_12amin_physical_units_from_real_Nico.npy --flat2hpx --nside 2048 --apodization-file /global/homes/j/jianyao/Small_Scale_Foreground/mask_320*320.npy --adaptive-reprojection
        
        srun -n 16 python reproject2fullsky_mpi.py --pixelsize 0.9375 --npix 1280 --overlap 2   --verbose  --flat-projection  /pscratch/sd/j/jianyao/forse_output/Nico_Q_20amin_20x20_1280.npy --flat2hpx --nside 4096 --apodization-file /global/homes/j/jianyao/Small_Scale_Foreground/mask_1280*1280.npy --adaptive-reprojection
        
        pass
    
    def power_spectra_patch(self, ):
        
        pass
    
    def power_spectra_full_sky(self, ):
        
        pass
    
    