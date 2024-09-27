from randomwave import *
from scipy.io import savemat

#### Meshgrid ####
n_grid = 128
x = np.linspace(-1,1,n_grid+1)
y = np.linspace(-1,1,n_grid+1)
z = np.linspace(-1,1,n_grid+1)

r_grid = np.meshgrid(x,y,z) 

#### Wave vector distribution ####
kz_list = np.random.rand(100)*10
kappa_list = np.exp(np.random.rand(100)*5)
alpha_list = np.random.rand(100)/2
sigma_f_list = np.random.rand(100)*9+1

n_wave = 60
n_sample = 100

S_q_kxy = []
for k, kappa, alpha, sigma_f in zip(kz_list,kappa_list,alpha_list,sigma_f_list):
    print(k, kappa, alpha, sigma_f)
    k_mean = np.array([0,0,20])*np.pi # lamellar perpendicular to z axis 
    k_var  = (np.array([0,0,k])*np.pi)**2
    k_cov  = np.diagflat(k_var)

    S_q_list = []

    for i in range(n_sample):
        # create sample
        rho = sample_wave_MO(r_grid,k_mean,k_cov,n_wave = n_wave,kappa = kappa)

        # calculate scattering function
        box_size = 2
        n_grid_scale = 256
        scale = n_grid_scale/rho.shape[0]
        dq = 2*np.pi/box_size
        qq = np.arange(n_grid_scale/2)*dq

        S_q_i = scatter_grid(rho,alpha,qq,scale=scale, fuzziness=sigma_f)
        S_q_list.append(S_q_i)

    S_q = np.mean(np.array(S_q_list),axis=0)
    S_q_kxy.append(S_q)

s_q_kxy = np.array(S_q_kxy)

#### save data ####
parameter_grid = np.array([k for k in zip(kz_list,kappa_list,alpha_list,sigma_f_list)])
filename = 'S_q_0_r.mat' #---------> kappa
mdict = {'S_q':S_q_kxy,'parameter_grid':parameter_grid}
savemat(filename,mdict)