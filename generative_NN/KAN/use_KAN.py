import torch
import torch.nn as nn
import yaml
import numpy as np
from kan import *
from use_training_set import *

# Default device
device = torch.device('cpu')
torch.set_default_dtype(torch.float32)

# Global model variable
model = None

def set_device(new_device):
    global device
    device = torch.device(new_device)
    print(f"Device set to: {device}")
    
def update_device(new_device):
    global device, x_train_torch, y_train_torch, Qx, Qx_inv, model
    set_device(new_device)
    x_train_torch = x_train_torch.to(device)
    y_train_torch = y_train_torch.to(device)
    Qx = Qx.to(device)
    Qx_inv = Qx_inv.to(device)
    if model is not None:
        model.to(device)
    print("All relevant tensors and models have been moved to the new device.")

def calculate_output_size(input_size, filter_size, padding, stride):
    return int((input_size - filter_size + 2*padding) / stride + 1)

def to_torch(array):
    return torch.from_numpy(array).float()

def to_torch_device(array, device=device):
    return torch.from_numpy(array.astype('float32')).float().to(device)

# Parameters for preparing the training data
# n_grid = 100
# x = np.linspace(-1, 1, n_grid + 1)
# y = np.linspace(-1, 1, n_grid + 1)
# z = np.linspace(-1, 1, n_grid + 1)

# r_grid = np.meshgrid(x, y, z)
# box_size = 2
# n_grid_scale = 256
# scale = n_grid_scale / r_grid[0].shape[0]
# dq = 2 * np.pi / box_size
# qq = np.arange(n_grid_scale / 2) * dq
# Q = qq[1:-1] / 20 / np.pi

# Load training data
config_file = 'setup_ts.txt'
x_train, y_train, Q_train = load_training_data(config_file)
x_train_torch = to_torch_device(x_train, device=device)
y_train_torch = to_torch_device(y_train, device=device)

# Add a column of ones to x to account for the bias term
x_train_bias = np.hstack([x_train, np.ones((x_train.shape[0], 1))])

# Initialize A and B
A = np.zeros((x_train.shape[1], y_train.shape[1]))
B = np.zeros(y_train.shape[1])

# For each target variable
for i in range(y_train.shape[1]):
    # Solve for A and B using least squares
    coef, _, _, _ = np.linalg.lstsq(x_train_bias, y_train[:, i], rcond=None)
    A[:, i] = coef[:-1]
    B[i] = coef[-1]

# the default Qx
f_Qx=lambda Q: np.vstack([Q,np.ones_like(Q)]).T
# Qx = np.vstack([Q_train,np.ones_like(Q_train)]).T
Qx = f_Qx(Q_train)
Qx_inv = to_torch_device((np.linalg.pinv(Qx)), device=device)
Qx = to_torch_device(Qx, device=device)

class SQ_KAN(nn.Module):
    def __init__(self, width, width_aug, grid, grid_aug, k, seed, device, grid_eps, noise_scale, base_fun, multiplier, Q_scale=1/4):
        super(SQ_KAN, self).__init__()
        self.kan_aug = KAN(width=width_aug, grid=grid_aug, k=k, seed=seed, device=device, grid_eps=grid_eps, noise_scale=noise_scale, base_fun=base_fun)
        # self.kan_aug.update_grid_from_samples(x_train_torch)
        self.kan = KAN(width=width, grid=grid, k=k, seed=seed, device=device, noise_scale=noise_scale, base_fun=base_fun)
        self.Q_torch = to_torch_device((Q_train - 2)*Q_scale, device=device)
        self.A = to_torch_device(A, device=device)
        self.B = to_torch_device(B, device=device)
        self.multiplier = multiplier
        self.Q_scale = Q_scale
        
    def forward(self, x):
        bg = (x @ self.A + self.B)
        # bg_expanded = bg.unsqueeze(1).expand(-1, self.Q_torch.size(0), -1)
        # Compute the mean of individual bg
        # bg_mean = bg_expanded.mean(dim=-1)
        # Linear approximation of the background
        Qab = Qx_inv@bg.T
        bg_lin = (Qx@Qab).T
        
        x = self.kan_aug(x)
        x_expanded = x.unsqueeze(1).expand(-1, self.Q_torch.size(0), -1)
        Q_expanded = self.Q_torch.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(-1))
        Q_params = torch.cat([Q_expanded, x_expanded], dim=-1)
        Q_params_reshaped = Q_params.view(-1, Q_params.size(-1))
        sq_full = self.kan(Q_params_reshaped)
        sq_full_reshaped = sq_full.view(x.size(0), self.Q_torch.size(0))
        return sq_full_reshaped*self.multiplier + bg_lin

def build_model(config, device=device):
    model = SQ_KAN(
        width=config['width'],
        width_aug=config['width_aug'],
        grid=config['grid'],
        grid_aug=config['grid_aug'],
        k=config['k'],
        seed=config['seed'],
        device=device,
        grid_eps=config['grid_eps'],
        noise_scale=config['noise_scale'],
        base_fun=config['base_fun'],
        multiplier=config['multiplier'],
        Q_scale=config['Q_scale']
    )
    return model

def f_IQ_KAN(model, x, Q, f_Qx):
    Qx_sample = to_torch_device(f_Qx(Q), device=device)
    
    # Transform x using kan_aug
    n_data = x.shape[0]
    x = x.view(-1, 3)
    # x[:,0] = np.exp(x[:,0])
    # x[:,1] = np.exp(x[:,1])
    x_transformed = model.kan_aug(x)
    
    # Transform Q using to_torch_device
    Q_scale = model.Q_scale
    Q_torch = to_torch_device((Q - 2)*Q_scale, device=device)
    
    # Calculate bg
    bg = (x @ model.A + model.B)
    # bg_expanded = bg.unsqueeze(1).expand(-1, self.Q_torch.size(0), -1)
    # Compute the mean of individual bg
    # bg_mean = bg_expanded.mean(dim=-1)
    # Linear approximation of the background
    Qab = Qx_inv@bg.T
    bg_lin = (Qx_sample@Qab).T
    
    # Expand dimensions to match Q_torch
    x_expanded = x_transformed.unsqueeze(1).expand(-1, Q_torch.size(0), -1)
    Q_expanded = Q_torch.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(-1))
    
    # Combine Q and x
    Q_params = torch.cat([Q_expanded, x_expanded], dim=-1)
    Q_params_reshaped = Q_params.view(-1, Q_params.size(-1))
    
    # Produce f(Q, x) using kan
    f_Q_x = model.kan(Q_params_reshaped)
    f_Q_x_reshaped = f_Q_x.view(x.size(0), Q_torch.size(0))
    
    # print(f_Q_x_reshaped.shape)
    # print(bg_mean)    
    # Add bg to the final output
    return f_Q_x_reshaped*model.multiplier + bg_lin

# cos model
class SQ_KAN_cos(nn.Module):
    def __init__(self, width, width_aug, grid, grid_aug, k, seed, device, grid_eps, noise_scale, base_fun, multiplier, Q_scale=1/4):
        super(SQ_KAN_cos, self).__init__()
        self.kan_aug = KAN(width=width_aug, grid=grid_aug, k=k, seed=seed, device=device, grid_eps=grid_eps, noise_scale=noise_scale, base_fun=base_fun)
        # self.kan_aug.update_grid_from_samples(x_train_torch)
        self.kan = KAN(width=width, grid=grid, k=k, seed=seed, device=device, noise_scale=noise_scale, base_fun=base_fun)
        self.Q_torch = to_torch_device(Q_train)
        self.Q_torch_scale = to_torch_device((Q_train-2)*Q_scale)
        self.A = to_torch_device(A)
        self.B = to_torch_device(B)
        self.multiplier = multiplier
        self.Q_scale = Q_scale
        
    def forward(self, x):
        bg = (x@self.A+self.B)
        # bg_expanded = bg.unsqueeze(1).expand(-1, self.Q_torch.size(0), -1)
        # Compute the mean of individual bg
        # bg_mean = bg_expanded.mean(dim=-1)
        # Linear approximation of the background
        Qab = Qx_inv@bg.T
        bg_lin = (Qx@Qab).T
        
        x = self.kan_aug(x)
        x_expanded = x.unsqueeze(1).expand(-1, self.Q_torch.size(0), -1)
        Q_expanded = self.Q_torch_scale.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(-1))
        Q_params = torch.cat([Q_expanded, x_expanded], dim=-1)
        Q_params_reshaped = Q_params.view(-1, Q_params.size(-1))
        
        G_full = self.kan(Q_params_reshaped)
        G_full_reshaped = G_full.view(x.size(0), self.Q_torch_scale.size(0), 2)  # (n_sample, n_Q, 2)
        
        output_1 = G_full_reshaped[:, :, 0]
        output_2 = G_full_reshaped[:, :, 1]
        
        sq_full = output_1*torch.cos(2*np.pi*self.Q_torch) + output_2
        sq_full_reshaped = sq_full.view(x.size(0), self.Q_torch.size(0))
        
        return sq_full_reshaped*self.multiplier + bg_lin

def build_model_cos(config, device=device):
    model = SQ_KAN_cos(
        width=config['width'],
        width_aug=config['width_aug'],
        grid=config['grid'],
        grid_aug=config['grid_aug'],
        k=config['k'],
        seed=config['seed'],
        device=device,
        grid_eps=config['grid_eps'],
        noise_scale=config['noise_scale'],
        base_fun=config['base_fun'],
        multiplier=config['multiplier'],
        Q_scale=config['Q_scale']
    )
    return model

def f_IQ_KAN_cos(model, x, Q, f_Qx):
    Qx_sample = to_torch_device(f_Qx(Q), device=device)
    
    # Transform x using kan_aug
    n_data = x.shape[0]
    x = x.view(-1, 3)
    # x[:,0] = np.exp(x[:,0])
    # x[:,1] = np.exp(x[:,1])
    x_transformed = model.kan_aug(x)
    
    # Transform Q using to_torch_device
    Q_scale = model.Q_scale
    Q_torch = to_torch_device((Q - 2)*Q_scale, device=device)
    
    # Calculate bg
    bg = (x @ model.A + model.B)
    # bg_expanded = bg.unsqueeze(1).expand(-1, self.Q_torch.size(0), -1)
    # Compute the mean of individual bg
    # bg_mean = bg_expanded.mean(dim=-1)
    # Linear approximation of the background
    Qab = Qx_inv@bg.T
    bg_lin = (Qx_sample@Qab).T
    
    # Expand dimensions to match Q_torch
    x_expanded = x_transformed.unsqueeze(1).expand(-1, Q_torch.size(0), -1)
    Q_expanded = Q_torch.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(-1))
    
    # Combine Q and x
    Q_params = torch.cat([Q_expanded, x_expanded], dim=-1)
    Q_params_reshaped = Q_params.view(-1, Q_params.size(-1))
    
    G_full = model.kan(Q_params_reshaped)
    G_full_reshaped = G_full.view(x.size(0), model.Q_torch_scale.size(0), 2)  # (n_sample, n_Q, 2)
    
    output_1 = G_full_reshaped[:, :, 0]
    output_2 = G_full_reshaped[:, :, 1]
    
    sq_full = output_1*torch.cos(2*np.pi*model.Q_torch) + output_2
    sq_full_reshaped = sq_full.view(x.size(0), model.Q_torch.size(0))
    # print(f_Q_x_reshaped.shape)
    # print(bg_mean)    
    # Add bg to the final output
    return sq_full_reshaped*model.multiplier + bg_lin

def update_Qx(f_Qx, config):
    global Qx_inv, model, f_IQ_KAN
    Qx_inv = to_torch_device(np.linalg.pinv(f_Qx(Q_train)), device=device)
    print("Qx and Qx_inv have been defined and updated.")
    # Redefine the model since Qx has been updated
    model = build_model(config['Model Setup'])
    print("Model has been redefined.")

def main():
    with open('setup_model.txt', 'r') as file:
        config = yaml.safe_load(file)
    
    global model
    model = build_model(config['Model Setup'])
    print(model)

if __name__ == "__main__":
    main()