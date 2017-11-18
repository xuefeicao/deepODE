import numpy as np 
import random
import scipy.sparse
import rpy2.robjects as robjects
from sklearn.datasets import make_sparse_spd_matrix
from numpy import linalg as LA
from scipy.integrate import simps
import matplotlib.pyplot as plt
#this is specific for stop and go fMRI data
def CDN_generate(seed,n_area=6,SNR=0,A_u=True,B_u=True,C_u=True,D_u=True):

    """ define a function to generate data from CDN model, please download precomp_0.Rdata 
    para: seed for random number
          n_area: dimension of the system
          SNR: noise added to the system ,i.e. SNR
          A_u: boolean variable for whether you want to update A
          B_u: ...
          C_u: ...
          D_u: ... 
    return: fmri data(dimension: n_area * length), stimuli(dimension: J * length)), A, B, C, D
    length means the length of time series
    """
    #load precomputed data
    robjects.r.load('precomp_0.rdata')
    t_1=np.array(robjects.r['x'][20])
    l_t_1=t_1.shape[0]
    hrf=np.array(robjects.r['x'][21])
    t=np.array(robjects.r['x'][23])
    Q4_1=np.array(robjects.r['x'][27])
    t_all=np.array(robjects.r['x'][28])
    t_U_1=np.array(robjects.r['x'][29])
    t_i_all=t_all
    t_i=t
    l_t_all=t_i_all.shape[0]
    row_n=int(np.array(robjects.r['x'][30])[0])
    l_t_0=row_n 
    J=int(np.array(robjects.r['x'][31])[0])
    dt=np.array(robjects.r['x'][34])[0]
    fold=np.array(robjects.r['x'][35])[0]


    #generate random parameters
    np.random.seed(seed)
    #-0.1-0.1
    #-0.1-0.1
    def para_gen(n_area):
        sparsity=np.random.uniform(0.1,0.7) 
        lower_b=np.random.uniform(-0.1,0.1)
        upper_b=np.random.uniform(lower_b,0.1)
        which_kind=np.random.rand()
        print which_kind
        if which_kind<1.0/3:
            tmp=-make_sparse_spd_matrix(n_area,1-sparsity,smallest_coef=lower_b,largest_coef=upper_b)
        elif which_kind<2.0/3:
            while True:
                lower_b=np.random.uniform(-0.1,0.1)
                upper_b=np.random.uniform(lower_b,0.1)
                tmp=np.random.uniform(-1,1,(n_area,n_area))
                tmp=tmp-np.sort(np.real(LA.eig(tmp)[0]))[-1]*np.eye(n_area)
                fmax=np.amax(tmp)
                fmin=np.amin(tmp)
                tmp=(upper_b-lower_b)/(fmax-fmin)*tmp+(lower_b*fmax-upper_b*fmin)/(fmax-fmin)
                if np.sort(np.real(LA.eig(tmp)[0]))[-1]<=0:
                    break
        else:
            tmp=np.random.uniform(-1,1,(n_area,n_area))
            tmp=(tmp-tmp.T)/2
            tmp=abs(lower_b)/np.amax(abs(tmp))*tmp
        return tmp

    A=para_gen(n_area)
    B=np.zeros((n_area,n_area,J))
    for j in range(J):
        B[:,:,j]=para_gen(n_area)
    C=np.random.uniform(-1,1,size=(n_area,J))
    D=np.random.uniform(-50,50,size=(n_area,1))

    def ODE_solve(A,B,C,D):
        h=fold*dt
        x_l=np.zeros((n_area,l_t_all))
        x_l[:,0]=D.reshape((-1))
        for i in range(1,l_t_all):
            tmp=0
            for j in range(J):
                tmp=tmp+Q4_1[j,i-1]*np.dot(B[:,:,j],x_l[:,i-1])
            k1=np.dot(A,x_l[:,i-1]) + tmp + np.dot(C,Q4_1[:,i-1])
            tmp=0
            for j in range(J):
                tmp=tmp+t_U_1[j,i-1]*np.dot(B[:,:,j],(x_l[:,i-1]+h/2*k1))
            k2=np.dot(A,(x_l[:,i-1]+h/2*k1))+ tmp+ np.dot(C,t_U_1[:,i-1])
            tmp=0
            for j in range(J):
                tmp=tmp+t_U_1[j,i-1]*np.dot(B[:,:,j],(x_l[:,i-1]+h/2*k2))
            k3=np.dot(A,(x_l[:,i-1]+h/2*k2))+ tmp+ np.dot(C,t_U_1[:,i-1])
            tmp=0
            for j in range(J):
                tmp=tmp+Q4_1[j,i]*np.dot(B[:,:,j],(x_l[:,i-1]+h*k3))
            k4=np.dot(A,(x_l[:,i-1]+h*k3))+ tmp+ np.dot(C,Q4_1[:,i])
            x_l[:,i]=x_l[:,i-1]+1.0*h/6*(k1+2*k2+2*k3+k4)
        return x_l

    def fmri_data(A,B,C,D,SNR):
        x_l=ODE_solve(A,B,C,D)
        z=np.zeros((n_area,l_t_0))
        for j in range(l_t_0):
            tmp=np.zeros((n_area,l_t_1))
            j_1=int(1/fold)*j+1
            in_1=min(j_1,l_t_1)
            if j_1-in_1-1>=0:
                tmp[:,0:in_1]=x_l[:,(j_1-1):(j_1-1-in_1):-1]
            else:
                tmp[:,0:in_1]=x_l[:,(j_1-1)::-1]
            for m in range(n_area):
                z[m,j]=simps(tmp[m,:]*hrf,t_1)
        sd=np.mean(abs(z))
        z=z+np.random.normal(0,sd*SNR,z.shape)
        return z

        
    return fmri_data(A,B,C,D,SNR), Q4_1[:,::int(1/fold)], A, B, C, D 

