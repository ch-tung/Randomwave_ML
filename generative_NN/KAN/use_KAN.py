import torch
import torch.nn as nn
import yaml
from kan import *
from use_training_set import *

# Default device
device = torch.device('cpu')
torch.set_default_dtype(torch.float32)

def set_device(new_device):
    global device
    device = torch.device(new_device)
    print(f"Device set to: {device}")

def calculate_output_size(input_size, filter_size, padding, stride):
    return int((input_size - filter_size + 2*padding) / stride + 1)

def to_torch(array):
    return torch.from_numpy(array).float()

def to_torch_device(array, device=device):
    return torch.from_numpy(array.astype('float32')).float().to(device)

# Parameters for preparing the training data
n_grid = 100
x = np.linspace(-1, 1, n_grid + 1)
y = np.linspace(-1, 1, n_grid + 1)
z = np.linspace(-1, 1, n_grid + 1)

r_grid = np.meshgrid(x, y, z)
box_size = 2
n_grid_scale = 256
scale = n_grid_scale / r_grid[0].shape[0]
dq = 2 * np.pi / box_size
qq = np.arange(n_grid_scale / 2) * dq
Q = qq[1:-1] / 20 / np.pi

# Load training data
config_file = 'setup_ts.txt'
x_train, y_train = load_training_data(config_file)

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

class SQ_KAN(nn.Module):
    def __init__(self, width, width_aug, grid, k, seed, device, grid_eps, noise_scale, base_fun):
        super(SQ_KAN, self).__init__()
        self.kan_aug = KAN(width=width_aug, grid=10, k=k, seed=seed, device=device, grid_eps=grid_eps, noise_scale=noise_scale, base_fun=base_fun)
        self.kan = KAN(width=width, grid=grid, k=k, seed=seed, device=device, noise_scale=noise_scale, base_fun=base_fun)
        self.Q_torch = to_torch_device((Q - 2) / 4, device=device)
        self.A = to_torch_device(A, device=device)
        self.B = to_torch_device(B, device=device)
        
    def forward(self, x):
        bg = (x @ self.A + self.B)
        bg_expanded = bg.unsqueeze(1).expand(-1, self.Q_torch.size(0), -1)
        # Compute the mean of individual bg
        bg_mean = bg_expanded.mean(dim=-1)
        
        x = self.kan_aug(x)
        x_expanded = x.unsqueeze(1).expand(-1, self.Q_torch.size(0), -1)
        Q_expanded = self.Q_torch.unsqueeze(0).unsqueeze(-1).expand(x.size(0), -1, x.size(-1))
        Q_params = torch.cat([Q_expanded, x_expanded], dim=-1)
        Q_params_reshaped = Q_params.view(-1, Q_params.size(-1))
        sq_full = self.kan(Q_params_reshaped)
        sq_full_reshaped = sq_full.view(x.size(0), self.Q_torch.size(0))
        return sq_full_reshaped + bg_mean

def build_model(config, device=device):
    model = SQ_KAN(
        width=config['width'],
        width_aug=config['width_aug'],
        grid=config['grid'],
        k=config['k'],
        seed=config['seed'],
        device=device,
        grid_eps=config['grid_eps'],
        noise_scale=config['noise_scale'],
        base_fun=config['base_fun']
    )
    return model

def main():
    with open('setup_model.txt', 'r') as file:
        config = yaml.safe_load(file)
    
    model = build_model(config['Model Setup'])
    print(model)

if __name__ == "__main__":
    main()