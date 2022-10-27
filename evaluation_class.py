import numpy as np


class evaluation(object):
    
    def __init__(self, data_dir, model_dir, mf_dir, checkpoint):
        
        self.model_dir = model_dir; self.checkpoint = checkpoint
        if isinstance(data_dir, str):
            Thr, Ls = np.load(data_dir)
        else:
            Thr, Ls = data_dir

        Ls_rescaled = np.zeros_like(Ls)
        for i in range(Ls.shape[0]):

                Ls_rescaled[i] = rescale_min_max(Ls[i])

        Ls_rescaled = Ls_rescaled.reshape((Ls.shape[0], Ls.shape[1], Ls.shape[1], 1)) 
        
        self.Ls_rescaled = Ls_rescaled
        
        rhos_t, f_t, u_t, chi_t = [], [], [], []
        
        npatches = 174
        for i in range(0,input_patches):

            mT = rescale_min_max(Thr[i], return_min_max=False)
            rhos_T, f_T, u_T, chi_T= get_functionals(mT)

            f_t.append(f_T);  u_t.append(u_T); chi_t.append(chi_T)

        f_t = np.array(f_t); u_t = np.array(u_t); chi_t = np.array(chi_t)
        
    def get_one_MF_only_NN(self, k):
        '''
        need to define checkpoint first
        '''
        checkpoint.restore(self.model_dir + 'training_checkpoints/ckpt-%s'%k)
        NNout = generator.predict(self.Ls_rescaled)

        rhos_nn, f_nn, u_nn, chi_nn = [], [], [], []  
        if k % 10 == 0:
            print(k)
        for i in range(0,input_patches):

            mNN = rescale_min_max(NNout[i,:,:,0], return_min_max=False)
            rhos_NN, f_NN, u_NN, chi_NN= get_functionals(mNN)

            f_nn.append(f_NN); u_nn.append(u_NN);chi_nn.append(chi_NN); 

        f_nn = np.array(f_nn); u_nn = np.array(u_nn); chi_nn = np.array(chi_nn); 

        m1_nnq = compute_intersection(rhos_T, 
                         [np.mean(f_t, axis=0)-np.std(f_t, axis=0), np.mean(f_t, axis=0)+np.std(f_t, axis=0)], 
                         [np.mean(f_nn, axis=0)-np.std(f_nn, axis=0),np.mean(f_nn, axis=0)+np.std(f_nn, axis=0)], 
                         npt=100000)
        m2_nnq = compute_intersection(rhos_T, 
                             [np.mean(u_t, axis=0)-np.std(u_t, axis=0), np.mean(u_t, axis=0)+np.std(u_t, axis=0)], 
                             [np.mean(u_nn, axis=0)-np.std(u_nn, axis=0),np.mean(u_nn, axis=0)+np.std(u_nn, axis=0)], 
                             npt=100000)
        m3_nnq = compute_intersection(rhos_T, 
                             [np.mean(chi_t, axis=0)-np.std(chi_t, axis=0), np.mean(chi_t, axis=0)+np.std(chi_t, axis=0)], 
                             [np.mean(chi_nn, axis=0)-np.std(chi_nn, axis=0),np.mean(chi_nn, axis=0)+np.std(chi_nn, axis=0)], 
                             npt=100000)
        return f_nn, u_nn, chi_nn, m1_nnq, m2_nnq, m3_nnq
        