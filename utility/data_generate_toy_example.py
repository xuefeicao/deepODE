import numpy as np 
import random
import scipy.sparse
import rpy2.robjects as robjects
from sklearn.datasets import make_sparse_spd_matrix
from numpy import linalg as LA
from scipy.integrate import simps
import matplotlib.pyplot as plt
import scipy.sparse as sp
#this is specific for stop and go fMRI data
def Toy_generate(seed, n_area=1, SNR=0, which_data=1, scale = 1, shifted = 1):

    """ define a function to generate data from CDN model, please download precomp_0.Rdata 
    para: seed for random number
          n_area: dimension of the system
          SNR: noise added to the system ,i.e. SNR 
          which_data: 1 (negative definite matrix), 2 (total random with negative eigenvalues), 3 (antisymmetric), -1 (default, random select one scenario)
          scale: scale of A, D
          shifted : 1, shifted time step 
          Note: for this toy example, which data must be 1 !!!
    return: fmri data(dimension: n_area * length), fmri_shifted(dimension: n_area * length)-padding 0,  A, D, x(the orginal data without convolution)
    length means the length of time series
    """
    #
    robjects.r.load('precomp_0.rdata')
    hrf = np.array(robjects.r['x'][21])
    t_1 = np.array(robjects.r['x'][20])
    fold = 0.25
    dt = 0.01
    row_n = 201
    l_t_all = (row_n - 1) * int(1 / fold) + 1
    l_t_0 = row_n
    l_t_1 = hrf.shape[0]


    #generate random parameters
    np.random.seed(seed)
    #-0.1-0.1
    #-0.1-0.1
    def para_gen(n_area):
        sparsity = np.random.uniform(0.1, 0.7) 
        lower_b = np.random.uniform(-0.1, 0.1) * scale 
        upper_b = np.random.uniform(lower_b, 0.1) * scale
       
        if which_data == 1:
            which_kind = 1.0 / 6
        elif which_data == 2:
            which_kind = 1.0 / 2
        elif which_data == 3:
            which_kind = 5.0 / 6
        else:
            which_kind = np.random.rand()
            
        def rand_1(n):
            return np.random.uniform(-1, 1, n)
        if which_kind < 1.0 / 3:
            tmp = -make_sparse_spd_matrix(n_area, 1 - sparsity, smallest_coef=lower_b, largest_coef=upper_b)
            if n_area == 1:
                tmp = - np.random.uniform(0.1,  1) * scale
	elif which_kind < 2.0 / 3:
            while True:
		lower_b = np.random.uniform(-0.1, 0.1)
		upper_b = np.random.uniform(lower_b, 0.1)
		tmp = np.random.uniform(-1, 1, (n_area,n_area))
                
		tmp = tmp - np.sort(np.real(LA.eig(tmp)[0]))[-1] * np.eye(n_area)
		fmax = np.amax(tmp)
		fmin = np.amin(tmp)
		tmp = (upper_b-lower_b) / (fmax - fmin) * tmp + (lower_b * fmax - upper_b * fmin) / (fmax - fmin)
                tmp = (abs(sp.random(n_area, n_area, density=sparsity).A) > 0) * tmp
		if np.sort(np.real(LA.eig(tmp)[0]))[-1] <= 0:
		    break
	else:
            #tmp = np.random.uniform(-1, 1, (n_area,n_area))
            tmp = sp.random(n_area, n_area, density=sparsity, data_rvs=rand_1).A
	    tmp = (tmp - tmp.T) / 2
	    tmp=abs(lower_b) / np.amax(abs(tmp)) * tmp
        return tmp

    A = np.zeros((n_area,n_area))
    D = np.zeros((n_area,1))


    A = para_gen(n_area)
    D = np.random.uniform(-50, 50, size=(n_area,1)) * scale

    def ODE_solve(A, D):
        h = fold * dt
        x_l = np.zeros((n_area,l_t_all))
        x_l[:,0] = D.reshape((-1))
        for i in range(1,l_t_all):
            x_l[:,i] = x_l[:,i-1] + h * np.dot(A, x_l[:,i-1]) 
        return x_l

    def fmri_data(A, D, SNR):
        x_l = ODE_solve(A, D)
        z = np.zeros((n_area,l_t_0))
        for j in range(l_t_0):
            tmp = np.zeros((n_area, l_t_1))
            j_1 = int(1 / fold) * j + 1
            in_1 = min(j_1, l_t_1)
            if j_1-in_1-1 >= 0:
                tmp[:,0:in_1] = x_l[:,(j_1-1):(j_1-1-in_1):-1]
            else:
                tmp[:,0:in_1] = x_l[:,(j_1-1)::-1]
            for m in range(n_area):
                z[m,j] = simps(tmp[m,:] * hrf, t_1)
        sd = np.mean(abs(z))
        z = z + np.random.normal(0, sd * SNR, z.shape)
        return z, x_l

    y, x= fmri_data(A, D, SNR)
    shifted_y = np.zeros(y.shape)
    shifted_y[:,0:(row_n-shifted)] = y[:,shifted::]


    return y, shifted_y, A, D, x

