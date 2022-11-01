import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import healpy as hp
import pymaster as nmt
import sys
# # sys.path.append('/global/cscratch1/sd/jianyao/ForSE/')
sys.path.append('/pscratch/sd/j/jianyao/forse_codes/') 

from forse.tools.img_tools import rescale_min_max
from forse.tools.mix_tools import get_functionals, compute_intersection

### load output (small scales only at 3amin) from the neural network 
# :shape: (174*49, 320, 320)

class post_training(object):
    '''
    All processes after the training.
    ss_I: intensity small scales 
    training_files_Q/U; input training files for the NN, shape:(2, 348, 320, 320)
    '''
    
    def __init__(self, NNout_Q, NNout_U, ss_I, Ls_Q_20amin, Ls_U_20amin, MF = True):
        
        self.NNout_Q = NNout_Q.reshape(174,49,320,320);
        self.NNout_U = NNout_U.reshape(174,49,320,320);
        
        self.thr = ss_I; # intensity small scales at 12amin or 3amin
        self.Ls_Q = Ls_Q_20amin.reshape(174,49,320,320)
        self.Ls_U = Ls_U_20amin.reshape(174,49,320,320)
        
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

    def normalization(self, gauss_ss_ps, gauss_ss_mean_std,  mask_path = 'mask_320*320.npy', lmin = 40*14, lmax = 3500):
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

        mask = np.load(mask_path)

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
                NNmapQ_corr[i,j]=((maps_out_3Q[i,j]-np.mean(maps_out_3Q[i,j]))/np.sqrt(np.mean(cl_NN_uncoupledQ[0][ell_s:]/gauss_ss_ps[0,i,j][0][ell_s:]))+ gauss_ss_mean_std[0][i, j])*self.Ls_Q[i, j] 
                NNmapU_corr[i,j]=((maps_out_3U[i,j]-np.mean(maps_out_3U[i,j]))/np.sqrt(np.mean(cl_NN_uncoupledU[0][ell_s:]/gauss_ss_ps[1,i,j][0][ell_s:]))+ gauss_ss_mean_std[2][i, j])*self.Ls_U[i, j]
        
        self.NNmapQ_corr, self.NNmapU_corr = NNmapQ_corr, NNmapU_corr
    
    def get_one_MF(self, input_maps, npatches = 348, patch_N = False):
        '''
        Defined for output at 3amin, [174,49, 320, 320] or for ordinary maps with shape [348, 320, 320]
        for nn output at 3amin, npatches = 174; for intensity small scales, npatch = 348;
        
        Returns
        -------
        rhos: threshold values, normally [-1, 1]
        f, u, chi : three kinds of MFs for each patch
        
        '''
        rhos, f_all, u_all, chi_all = [], [], [], []
        
        if patch_N:
            assert npatches == 348;
            i_s = patch_N*2 - 1; i_e = (patch_N+1)*2 - 1
            maps_MF = input_maps[:, i_s:i_e, :, :].reshape(348, 320, 320) # for NN output with shape (174, 49, 320, 320)
            
        else: 
            maps_MF = input_maps # for intensity small scales with shape (348, 320, 320)
        
        for i in range(0,npatches):
    
            mT = rescale_min_max(maps_MF[i], return_min_max=False)
            rhos, f, u, chi= get_functionals(mT)

            f_all.append(f);  u_all.append(u); chi_all.append(chi)

        f_all = np.array(f_all); u_all = np.array(u_all); chi_all = np.array(chi_all)
        
        return rhos, f_all, u_all, chi_all
    
    def compute_overlapping(self, MF_P):
        '''
        compute the MFs overlapping fraction w.r.t. small scales of intensity maps
        '''
        
        rhos_T, f_t, u_t, chi_t = self.MF_I;
        rho_NN, f_nn, u_nn, chi_nn = MF_P;
        
        m1_nnq = compute_intersection(rhos_T, 
                         [np.mean(f_t, axis=0)-np.std(f_t, axis=0), np.mean(f_t, axis=0)+np.std(f_t, axis=0)], 
                         [np.mean(f_nn, axis=0)-np.std(f_nn, axis=0),np.mean(f_nn, axis=0)+np.std(f_nn, axis=0)], npt=100000)
        m2_nnq = compute_intersection(rhos_T, 
                         [np.mean(u_t, axis=0)-np.std(u_t, axis=0), np.mean(u_t, axis=0)+np.std(u_t, axis=0)], 
                         [np.mean(u_nn, axis=0)-np.std(u_nn, axis=0),np.mean(u_nn, axis=0)+np.std(u_nn, axis=0)], npt=100000)
        m3_nnq = compute_intersection(rhos_T, 
                         [np.mean(chi_t, axis=0)-np.std(chi_t, axis=0), np.mean(chi_t, axis=0)+np.std(chi_t, axis=0)], 
                         [np.mean(chi_nn, axis=0)-np.std(chi_nn, axis=0),np.mean(chi_nn, axis=0)+np.std(chi_nn, axis=0)], npt=100000)
        
        return m1_nnq, m2_nnq,m3_nnq
    
    def plot_MF(self, patch_N, savedir = False):
        
        rhos_Y, f_t, u_t, chi_t = self.MF_I
        MF_Q = self.get_one_MF(self.NNout_Q, patch_N = patch_N);
        MF_U = self.get_one_MF(self.NNout_U, patch_N = patch_N);
        
        results = [];
        results.append(self.compute_overlapping(MF_Q))
        results.append(self.compute_overlapping(MF_U))
        
        rhos_Y, f_nn_Q, u_nn_Q, chi_nn_Q = MF_Q
        rhos_Y, f_nn_U, u_nn_U, chi_nn_U = MF_U
        
        f_nn_all = [[f_nn_Q, u_nn_Q, chi_nn_Q],[f_nn_Q, u_nn_Q, chi_nn_Q]]
        f_i = [f_t, u_t, chi_t]
        fig, axes = plt.subplots(2,3, figsize=(24, 10))
        S = ['Q', 'U']
        for i in range(3):
            for j in range(2):
                f_nn = f_nn_all[j][i]; f_t = f_i[i];
                
                axes[j, i].fill_between(rhos_Y, 
                                     np.mean(f_nn, axis=0)-np.std(f_nn, axis=0), 
                                     np.mean(f_nn, axis=0)+np.std(f_nn, axis=0), 
                                     lw=1, label=r'$m_{ss}^{NN, %s}$'%S[j], alpha=0.5, color='#F87217')
                axes[j, i].plot(rhos_Y, np.mean(f_nn, axis=0), lw=3, ls='--', color='#D04A00')
                axes[j, i].fill_between(rhos_Y, 
                                     np.mean(f_t, axis=0)-np.std(f_t, axis=0), 
                                     np.mean(f_t, axis=0)+np.std(f_t, axis=0), 
                                     lw=2, label = r'$m_{ss}^{real, I}$', edgecolor='black', facecolor='None')
                axes[j, i].plot(rhos_Y, np.mean(f_t, axis=0), lw=2, ls='--', color='black')
                axes[j, i].set_ylabel(r'$\mathcal{V}_{%s}(\rho$) %s'%(i, S[j]), fontsize=20)
                axes[j, i].set_title('%.2f'%results[j][i], fontsize = 20)
                if i == 0:
                    axes[j, i].legend(fontsize = 25)
                if j == 1:
                    axes[j, i].set_xlabel(r'$\rho$', fontsize=20)
        if savedir:
            plt.savefig(savedir, format = 'pdf')
    
    def combine_to_20by20(self, save_dir):
        '''
        Recompose 5°x5° maps together to form 20°x20° maps
        '''
        # Create apodization masks
        x = np.cos(np.concatenate((np.arange(160)[::-1],np.arange(160)))*np.pi/(2*159))*np.cos(np.concatenate((np.arange(160)[::-1],np.arange(160)))*np.pi/(2*159))
        mask_cc = np.zeros((320,320))
        for i in range(320):
            for j in range (320):
                mask_cc[i,j] = x[i]*x[j] 

        mask_nw = np.copy(mask_cc)
        for i in range(160):
            for j in range(160):
                mask_nw[i,j] = 1
        for i in range(160):
            for j in range(160,320):
                mask_nw[i,j] = x[j]
        for i in range(160,320):
            for j in range(160):
                mask_nw[i,j] = x[i]

        mask_sw = np.rot90(mask_nw); mask_se = np.rot90(mask_sw); mask_ne = np.rot90(mask_se)

        mask_nn = np.copy(mask_cc)
        for i in range(160):
            for j in range(320): 
                mask_nn[i,j] = x[j]

        mask_ww = np.rot90(mask_nn); mask_ss = np.rot90(mask_ww); mask_ee = np.rot90(mask_ss)
        
        # Apply apodization masks to 5°x5° maps
        maps_msk_3Q = np.zeros(np.shape(self.NNmapQ_corr));
        maps_msk_3U = np.zeros(np.shape(self.NNmapU_corr));
        maps_ren2_3Q = self.NNmapQ_corr;
        maps_ren2_3U = self.NNmapU_corr;

        for i in range(174):  

            # angle masks
            maps_msk_3Q[i,0,:,:] = maps_ren2_3Q[i,0,:,:]*mask_nw
            maps_msk_3Q[i,6,:,:] = maps_ren2_3Q[i,6,:,:]*mask_ne
            maps_msk_3Q[i,42,:,:] = maps_ren2_3Q[i,42,:,:]*mask_sw
            maps_msk_3Q[i,48,:,:] = maps_ren2_3Q[i,48,:,:]*mask_se

            maps_msk_3U[i,0,:,:] = maps_ren2_3U[i,0,:,:]*mask_nw
            maps_msk_3U[i,6,:,:] = maps_ren2_3U[i,6,:,:]*mask_ne
            maps_msk_3U[i,42,:,:] = maps_ren2_3U[i,42,:,:]*mask_sw
            maps_msk_3U[i,48,:,:] = maps_ren2_3U[i,48,:,:]*mask_se

            # side masks
            for j in range(1,6):
                maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_nn
                maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_nn
            for j in range(43,48):
                maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_ss
                maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_ss
            for j in [7,14,21,28,35]:
                maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_ww
                maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_ww
            for j in [13,20,27,34,41]:
                maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_ee
                maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_ee

            # center masks
            for j in range(8,13):
                maps_msk_3Q[i,j,:,:] = maps_ren2_3Q[i,j,:,:]*mask_cc
                maps_msk_3Q[i,j+7,:,:] = maps_ren2_3Q[i,j+7,:,:]*mask_cc
                maps_msk_3Q[i,j+14,:,:] = maps_ren2_3Q[i,j+14,:,:]*mask_cc
                maps_msk_3Q[i,j+21,:,:] = maps_ren2_3Q[i,j+21,:,:]*mask_cc
                maps_msk_3Q[i,j+28,:,:] = maps_ren2_3Q[i,j+28,:,:]*mask_cc

                maps_msk_3U[i,j,:,:] = maps_ren2_3U[i,j,:,:]*mask_cc
                maps_msk_3U[i,j+7,:,:] = maps_ren2_3U[i,j+7,:,:]*mask_cc
                maps_msk_3U[i,j+14,:,:] = maps_ren2_3U[i,j+14,:,:]*mask_cc
                maps_msk_3U[i,j+21,:,:] = maps_ren2_3U[i,j+21,:,:]*mask_cc
                maps_msk_3U[i,j+28,:,:] = maps_ren2_3U[i,j+28,:,:]*mask_cc

        # Recompose 20°x20° maps together
        maps_big_3Q = np.zeros([174,1280,1280])
        maps_big_3U = np.zeros([174,1280,1280])

        for i in range(174): 
            for j in range(0,1120,160):
                for k in range(0,1120,160):
                    maps_big_3Q[i,j:(j+320),k:(k+320)] += maps_msk_3Q[i,int(j/160)*7+int(k/160),:,:]
                    maps_big_3U[i,j:(j+320),k:(k+320)] += maps_msk_3U[i,int(j/160)*7+int(k/160),:,:]

        self.NN_20by20_Q = maps_big_3Q;
        self.NN_20by20_U = maps_big_3U;
        
        if save_dir:
            np.save(save_dir[0], maps_big_3Q)
            np.save(save_dir[1], maps_big_3U)

    def plot_maps_modify(self, Nico_20amin, maps_out_3_348, NNmap_corr_348, m = 36, n = 4, savedir = False):

        '''
        map visualization; maps at 20 amin; output from NN; renormalize the NN output and combine with the large scales
        m: sky_position. 0-174
        n: patch_position in the 7*7 square
        '''
        
        fig, axes = plt.subplots(3, 3, figsize = (12, 12))

        for l in range(3):
            if l == 0:
                axes[l][0].set_title(r'$M_{LS}$')
                axes[l][1].set_title(r'$M_{SS}$')
                axes[l][2].set_title(r'$M_{LS} + M_{SS}$')
            min_v = np.min(Nico_20amin[m+l, n]); max_v = np.max(Nico_20amin[m+l, n])
            axes[l][0].imshow(Nico_20amin[m+l, n], vmin = min_v, vmax = max_v)
            axes[l][1].imshow(maps_out_3_348[m+l, n])
            im = axes[l][2].imshow(NNmap_corr_348[m+l, n], vmin = min_v, vmax = max_v)

            cax = fig.add_axes([0.92, 0.654 - l*0.267, 0.02, 0.234])
            fig.colorbar(im,cax = cax, ticks = [round(min_v), round(max_v)], extend = 'both', extendfrac = [0.1,0.05], extendrect = True)
            cax.yaxis.set_ticks_position('right')

            plt.text(1.25, 0.5, r'$\mu$K', rotation='vertical', transform=axes[l][2].transAxes)
        if savedir:
            plt.savefig(savedir, format = 'pdf')
    
    def plot_external(self, center, reso, maps_80amin, maps_12amin, maps_3amin, unify, savedir=False):
        '''
        center = [0,50]
        unify: use the min and max of maps at 12amin as the lower and upper limit for plot
        '''
        
        plt.figure(figsize = (8, 7.5))
        if unify:
            Q_img = hp.gnomview(maps_12amin[0], rot = center, reso=reso, xsize=320, no_plot=True, return_projected_map=True)
            U_img = hp.gnomview(maps_12amin[1], rot = center, reso=reso, xsize=320, no_plot=True, return_projected_map=True)
            hp.gnomview(maps_3amin[0], rot = center, reso = reso, xsize = 320, sub = 233, title = 'Q at 3 amin', min = np.min(Q_img), max = np.max(Q_img))
            hp.gnomview(maps_3amin[1], rot = center, reso = reso, xsize = 320, sub = 236, title = 'U at 3 amin', min = np.min(U_img), max = np.max(U_img))
            
        else:
            hp.gnomview(maps_3amin[0], rot = center, reso = reso, xsize = 320, sub = 233, title = 'Q at 3 amin')
            hp.gnomview(maps_3amin[1], rot = center, reso = reso, xsize = 320, sub = 236, title = 'U at 3 amin')            
            
        hp.gnomview(maps_80amin[0], rot = center, reso = reso, xsize = 320, sub = 231, title = 'Q at 80amin')
        hp.gnomview(maps_12amin[0], rot = center, reso = reso, xsize = 320, sub = 232, title = 'Q at 12amin')
        
        hp.gnomview(maps_80amin[1], rot = center, reso = reso, xsize = 320, sub = 234, title = 'U at 80amin')
        hp.gnomview(maps_12amin[1], rot = center, reso = reso, xsize = 320, sub = 235, title = 'U at 12amin')
              
        
        if savedir:
            plt.savefig(savedir, format = 'pdf')
        
    def reproject_to_fullsky(self, ):
        
        pass
    
    def load_w22(self, w22_file, npix, mask, lmax):
                       
        w22 = nmt.NmtWorkspaceFlat()
        try:
            w22.read_from(w22_file)
            print('weights loaded from %s' % w22_file)
            
        except: 
            l0_bins = np.arange(20, lmax, 40); lf_bins = np.arange(20, lmax, 40)+39
            b = nmt.NmtBinFlat(l0_bins, lf_bins) 
            f_2 = nmt.NmtFieldFlat(Lx, Ly, mask, [np.zeros((npix, npix)), np.zeros((npix, npix))], purify_b=True)
            w22.compute_coupling_matrix(f2, f2, b)
            w22.write_to(w22_file + '%s'%lmax)
            print('weights writing to disk')
            
        return w22
    
    def power_spectra_patch(self, Ls_80Q, Ls_80U, Ls_12Q, Ls_12U, GaussQ, GaussU, NN_combined_Q, NN_combined_U, N, lmax, mask_320, mask_1280, w22_320, w22_1280, save_dir = False):
        
        '''
        plot EE/BB power spectra for each flat patch with area 20x20deg2 and shape (174, 1280, 1280), which is recomposed from 5x5deg^2 patches. 
        For Large scales only, Large scales with gaussian small scales; Large scales with ForSE small scales.     
         w22_file = "w22_flat_1280_1280.fits"
         mask_path = 'mask_1280*1280.npy'
        '''
        
        Lx = np.radians(20.); Ly = np.radians(20.)        
        mask_320 = np.load(mask_320); mask_1280 = np.load(mask_1280)
        
        l0_bins = np.arange(20, lmax, 40); lf_bins = np.arange(20, lmax, 40)+39
        b = nmt.NmtBinFlat(l0_bins, lf_bins)
        b_320 = nmt.NmtBinFlat(np.arange(20, 1000, 40), np.arange(20, 1000, 40)+39)
        ells_uncoupled = b.get_effective_ells()
        ells_uncoupled_320 = b_320.get_effective_ells()
      
        Qmaps = [Ls_80Q[N], Ls_12Q[N], GaussQ[N], NN_combined_Q[N]];
        Umaps = [Ls_80U[N], Ls_12Q[N], GaussU[N], NN_combined_U[N]];
        w22_320 = self.load_w22(w22_320, 320, mask_320, lmax = 1000); w22_1280 = self.load_w22(w22_1280, 1280, mask_1280, lmax)
        
        cls_all = []; masks = [mask_320, mask_1280]; w22s = [w22_320, w22_1280]; bs = [b_320, b]
        
        for i in range(4):
            f_NN = nmt.NmtFieldFlat(Lx, Ly, masks[i//2], [Qmaps[i], Umaps[i]], purify_b=True) 
            cl_NN_coupled = nmt.compute_coupled_cell_flat(f_NN, f_NN, bs[i//2])
            w22 = w22s[i//2]
            cl_NN_uncoupled = w22.decouple_cell(cl_NN_coupled)     
            cls_all.append(cl_NN_uncoupled)        
    
        fig, axes = plt.subplots(1,2, figsize=(17, 5.5))                  
        names = ['EE', 'BB']
        for i in range(2):
            axes[i].loglog(ells_uncoupled_320, cls_all[0][i*3],  '--', lw=2, color='Black', alpha=0.5, label = 'GNILC 80 amin')
            axes[i].loglog(ells_uncoupled_320, cls_all[1][i*3],  '-', lw=3, color='Black', alpha=0.6, label = 'GNILC+NN 12 amin')
            axes[i].loglog(ells_uncoupled, cls_all[2][i*3], '-', label='GNILC+Gauss 3 amin', lw=4, color='#569A62', alpha=0.7)
            axes[i].loglog(ells_uncoupled, cls_all[3][i*3], '-', label='GNILC+NN 3 amin', lw=4, color='#F87217', alpha=0.7)
            axes[i].set_ylim(1e-6, 2e-1)
            axes[i].set_xticks([40, 100, 400, 1000])
            axes[i].set_title('%s'%names[i], fontsize=18)
            axes[i].set_xlabel(r'Multipole $\ell$', fontsize=18)
            axes[i].set_ylabel(r'$C_\ell$ [$\mu K^2$]', fontsize=18)
        axes[0].legend(fontsize = 15)
        if save_dir:
            plt.savefig(save_dir, format = 'pdf')
    
    def cl_anafast(self, map_QU, lmax):
        '''
        Return the full-sky power spetra, except monopole and dipole
        '''
        
        map_I = np.ones_like(map_QU[0])
        maps = np.row_stack((map_I, map_QU))
        cls_all = hp.anafast(maps, lmax = lmax)
        ells = np.arange(lmax+1)
        
        return ells[2:], cls_all[:, 2:]
    
    def cl_nmt(self, nside, msk_apo, map_QU, lmax, nlbins, w22_file = 'w22_2048_full_sky.fits'):
        '''
        nside:
        msk_apo: apodized mask
        nlbins: ell-number in each bin
        '''

        binning = nmt.NmtBin(nside=nside, nlb=nlbins, lmax=lmax, is_Dell=False)
        f2 = nmt.NmtField(msk_apo, [map_QU[0], map_QU[1]], purify_b=True)

        w22 = nmt.NmtWorkspace()
        try:
            w22.read_from(w22_file)
            print('weights loaded from %s' % w22_file)
        except:
            w22.compute_coupling_matrix(f2, f2, binning)
            w22.write_to(w22_file)
            print('weights writing to disk')

        cl22 = nmt.compute_full_master(f2, f2, binning, workspace = w22)

        return binning.get_effective_ells(), cl22
    
    def plot_spectra(self, cls_all, names, save_dir):
        '''
        cls_80p_80amin = {'ells':ell_80p_80amin, 'spectra':cl_80p_80amin, 'color':'r-', 'label':'80amin'}
        cls_80p_12amin = {'ells':ell_80p_12amin, 'spectra':cl_80p_12amin, 'color':'g-', 'label':'12amin'}
        '''
        names = ['EE', 'BB']
        fig, axes = plt.subplots(1,2, figsize = (30, 10))

        for i in range(len(cls_all)):
            ells = cls_all[i]['ells']; cl = cls_all[i]['spectra']; color = cls_all[i]['color']; label = cls_all[i]['label']
            axes[0].loglog(ells, abs(cl[0]), color, label = label)
            axes[1].loglog(ells, abs(cl[3]), color)

        line1 = Line2D([],[],linestyle='-', color='r')
        line2 = Line2D([],[],linestyle='-.', color='r')
        line3 = Line2D([],[],linestyle='--', color='r')
        axes[0].legend(fontsize = 25, loc = 'lower left')
        axes[1].legend([line1, line2, line3],['full sky', '80% sky', '40% sky'], fontsize = 25, loc = 'lower left')

        for j in range(2):
            axes[j].set_ylim(1e-9, 1e3)
            axes[j].set_title('%s'%names[j], fontsize = 25)
            axes[j].set_ylabel(r'C$\ell$', fontsize = 25)
            axes[j].set_xlabel(r'$\ell$', fontsize = 25)

        if save_dir:
            plt.savefig(save_dir, format = 'pdf')  