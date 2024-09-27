from randomwave_2level import *
from scipy.io import savemat

#### Meshgrid ####
n_grid = 256
x = np.linspace(-1,1,n_grid+1)
y = np.linspace(-1,1,n_grid+1)
z = np.linspace(-1,1,n_grid+1)

r_grid = np.meshgrid(x,y,z) 

#### Wave vector distribution ####
kz_list = (np.arange(10)+1)/10*2.5 #---------> k_z
cl_list = 0.25 + (np.arange(11))/10*0.75 #---------> cl
kz_grid, cl_grid = np.meshgrid(kz_list,cl_list)

n_wave = 60
n_sample = 50

#### kappa ####
kappa = 2**0/4 #---------> kappa

S_q_kxy = []
for k, cl in zip(kz_grid.flatten(),cl_grid.flatten()):
    print(k,cl,kappa)
    k_mean = np.array([0,0,10])*np.pi # lamellar perpendicular to z axis 
    k_var  = (np.array([0,0,k])*np.pi)**2
    k_cov  = np.diagflat(k_var)

    S_q_list = []

    for i in range(n_sample):
        # create sample
        rho = sample_wave_MO(r_grid,k_mean,k_cov,n_wave = n_wave,kappa = kappa)

        # calculate scattering function
        box_size = 2
        n_grid_scale = 512
        scale = n_grid_scale/rho.shape[0]
        dq = 2*np.pi/box_size
        qq = np.arange(n_grid_scale/4)*dq

        S_q_i = scatter_grid(rho,cl,qq,scale=scale)
        S_q_list.append(S_q_i)

    S_q = np.mean(np.array(S_q_list),axis=0)
    S_q_kxy.append(S_q)

s_q_kxy = np.array(S_q_kxy)

#### save data ####
k_grid = np.c_[np.array([k for k in zip(kz_grid.flatten(),cl_grid.flatten())]),
               np.ones(len(kz_grid.flatten()))*kappa]
filename = 'S_q_0.mat' #---------> kappa
mdict = {'S_q_kxy':S_q_kxy,'k_grid':k_grid}
savemat(filename,mdict)