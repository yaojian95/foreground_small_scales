import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt


### load output (small scales only at 3amin) from the neural network 
# :shape: (174*49, 320, 320)

class post_training(object):
    
    def __init__(self, NNout_Q, NNout_U, intensity_ss, MF = True):
        '''
        intensity_ss: small scales of intensity at 12'; shape:(348, 320, 320)
        '''
        
        self.NNout_Q = NNout_Q;
        self.NNout_U = NNout_U;
        self.thr = intensity_ss;
        
        if MF:
            self.MF_I = self.get_one_MF(self.thr, npatches = 348, patch_N = False)

    def first_normalization(self, gauss_ss_mean_std):
        '''
        Normalize the small scales w.r.t. Gaussian case in the map level;
        
        Parameters
        -----------------------------------------------------------------
        maps_out_3Q/U: small scales generated from the Neural Network. With shape: (174, 49, 320, 320);
        gauss_ss_mean_std: mean and std for each patch of small scales of Gaussian realization, defined by the ratio: 
        Gaussian_maps_3amin/Gaussian_maps_20amin; 4 in 1: Q_mean, Q_std, U_mean, U_std. With shape (4, 174, 49).
        
        Returns
        -------
        normalized maps.

        '''
        maps_out_3Q, maps_out_3U = self.NNout_Q, self.NNout_U
        for i in range(174):
            for j in range(49):
                maps_out_3Q[i, j] = (maps_out_3Q[i, j] - np.mean(maps_out_3Q[i, j]))/np.std(maps_out_3Q[i, j])*gauss_ss_mean_std[1][i, j] + gauss_ss_mean_std[0][i, j]
                maps_out_3U[i, j] = (maps_out_3U[i, j] - np.mean(maps_out_3U[i, j]))/np.std(maps_out_3U[i, j])*gauss_ss_mean_std[3][i, j] + gauss_ss_mean_std[2][i, j]

        return maps_out_3Q, maps_out_3U

    def normalization(self, gauss_ss_ps, gauss_ss_mean_std, Nico_20amin_Q, Nico_20amin_U, lmin = 40*14, lmax = 3500):
        '''
        Normalize the small scales w.r.t. Gaussian case in the power spectra level and multiply with the large scales to get a full-resolution maps, after the first normalization.
        
        Parameters
        ----------
        maps_out_3Q/U: small scales after the first normalization; With shape: (174, 49, 320, 320);
        gauss_ss_ps: power spectra for each patch of small scales of Gaussian realization; 2 in 1: cl_QQ and cl_UU; with shape: (2, 174, 49, 1, 25).
        Nico_20amin_Q/U: large scales, same as the input for the training; with shape (174,49,320,320).
        
        Returns
        -------
        
        full resolution maps with physical units.
        '''
        maps_out_3Q, maps_out_3U = self.first_normalization(gauss_ss_mean_std)
        
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
        
        self.NNmapQ_corr, self.NNmapU_corr = NNmapQ_corr, NNmapU_corr
        return NNmapQ_corr, NNmapU_corr
    
    def get_one_MF(self, input_maps, npatches = 348, patch_N = False):
        '''
        Defined for output at 3amin, [174,49, 320, 320] or for ordinary maps with shape [348, 320, 320]
        for nn output at 3amin, npatches = 174; for intensity small scales, npatch = 348;
        
        Returns
        -------
        rhos: threshold values, normally [-1, 1]
        f, u, chi : three kinds of MFs for each patch
        
        '''
        rhos, f, u, chi = [], [], [], []
        
        if patch_N:
            assert npatches == 174;
            i_s = patch_N*2 - 1; i_e = (patch_N+1)*2 - 1
            maps_MF = input_maps[:, i_s:i_e, :, :].reshape(348, 320, 320) # for NN output with shape (174, 49, 320, 320)
            
        else: 
            maps_MF = input_maps # for intensity small scales with shape (348, 320, 320)
        
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
    
    def combine_to_20by20(self, ):
        
        pass
    
    def visualization(self, ):
        
        pass
    
    
    def reproject_to_fullsky(self, ):
        
        pass
    
    def power_spectra_patch(self, ):
        
        pass
    
    def power_spectra_full_sky(self, ):
        
        pass
    
    