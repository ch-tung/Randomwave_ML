import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d as gf1d
import yaml

def load_training_data(config_file, extend=False, sm=True):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    
    n_grid = config['n_grid']
    n_grid_scale = config['n_grid_scale']
    box_size = config['box_size']
    dq = 2 * np.pi / box_size
    qq = np.arange(n_grid_scale / 2) * dq
    Q_train = qq[1:-1] / 20 / np.pi

    S_q_Grid = []
    p_Grid = []
    for i in range(10):
        filename = config['grid_ext'].format(i)
        data = loadmat(filename)
        k_grid = data['k_grid']
        S_q_kxy = data['S_q_kxy']
        S_q_Grid.append(S_q_kxy)
        p_Grid.append(k_grid)

    for i in range(10):
        filename = config['grid_ext_half'].format(i)
        data = loadmat(filename)
        k_grid = data['k_grid']
        S_q_kxy = data['S_q_kxy']
        S_q_Grid.append(S_q_kxy)
        p_Grid.append(k_grid)
        
    if extend:
        for i in range(10):
            filename = config['grid_ex_monoc'].format(i)
            data = loadmat(filename)
            k_grid = data['k_grid']
            S_q_kxy = data['S_q_kxy']
            S_q_Grid.append(S_q_kxy)
            p_Grid.append(k_grid)
            
        for i in range(10):
            filename = config['grid_ex_monoc_half'].format(i)
            data = loadmat(filename)
            k_grid = data['k_grid']
            S_q_kxy = data['S_q_kxy']
            S_q_Grid.append(S_q_kxy)
            p_Grid.append(k_grid)

    S_q_Grid = np.array(S_q_Grid).reshape(100 * len(p_Grid), 128)[:, 1:-1].astype(np.float32)
    p_Grid = np.array(p_Grid).reshape(100 * len(p_Grid), 3)
    S_q_sm_Grid = np.exp(np.array([gf1d(f, 1, mode='nearest') for f in np.log(S_q_Grid)])).astype(np.float32)
    log_S_q_Grid = np.log(S_q_Grid)
    log_S_q_sm_Grid = np.log(S_q_sm_Grid)

    k_z = p_Grid[:, 0].astype(np.float32)
    logk_z = np.log(k_z).astype(np.float32)
    alpha = p_Grid[:, 1].astype(np.float32)
    kappa = p_Grid[:, 2].astype(np.float32)
    logkappa = np.log(kappa).astype(np.float32)

    print('\nk_z in')
    print(np.unique(k_z))
    print('\nalpha in')
    print(np.unique(alpha))
    print('\nkappa in')
    print(np.flip(np.unique(kappa)))

    parameters_zscore = config['parameters_zscore']
    if parameters_zscore:
        k_z_mean = np.mean(logk_z)
        k_z_std = np.std(logk_z)
        k_z_z = (logk_z - k_z_mean) / k_z_std

        kappa_mean = np.mean(logkappa)
        kappa_std = np.std(logkappa)
        kappa_z = (logkappa - kappa_mean) / kappa_std

        alpha_mean = np.mean(alpha)
        alpha_std = np.std(alpha)
        alpha_z = (alpha - alpha_mean) / alpha_std

        parameters_train = np.array([k_z_z, kappa_z, alpha_z]).T
    else:
        parameters_train = np.array([logk_z, logkappa, alpha]).T

    y_train = log_S_q_Grid
    x_train = parameters_train
    
    if sm:
        y_train = log_S_q_sm_Grid

    return x_train, y_train, Q_train

# Example usage
# config_file = 'setup_model.txt'
# x_train, y_train = load_training_data(config_file)
# print(x_train.shape)
# print(y_train.shape)