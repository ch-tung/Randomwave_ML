import numpy as np
import scipy.interpolate as interp

import tensorflow as tf
print('tensorflow version = '+tf.__version__)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

k_z_list = np.arange(10)+1
kappa_list = 2**(np.arange(20)/2-2.5)
alpha_list = np.arange(10)/20
logkappa_list = np.log(kappa_list)

k_z_mean = np.mean(k_z_list)
k_z_std = np.std(k_z_list)
kappa_mean = np.mean(logkappa_list)
kappa_std = np.std(logkappa_list)
alpha_mean = np.mean(alpha_list)
alpha_std = np.std(alpha_list)

box_size = 2
n_grid_scale = 256
dq = 2*np.pi/box_size
qq = np.arange(n_grid_scale/2)*dq

#%% Load generative NN
## Transform the input to tensorflow tensor
def to_tf(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return arg

# -20 to 0
f_shift = -20
f_scale = 20

def f_inp(sq):
    return (sq-f_shift)/f_scale

def f_out(sq_pred):
    return (sq_pred*f_scale)+f_shift

n_conv = 48
n_kernel = 5
class VAE(tf.keras.Model):
    def __init__(self, latent_dim, sq_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        regularizer = tf.keras.regularizers.L2(1.0)
        self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(sq_dim)),
            tf.keras.layers.Reshape((sq_dim,1)),
            tf.keras.layers.Conv1D(
                filters=n_conv, kernel_size=n_kernel, strides=2, activation='relu',
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer,
                name='conv1d_en'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                latent_dim + latent_dim,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer,
                name='dense_en'),
        ]
        )

        self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(
                int(sq_dim/2)*n_conv, activation=tf.nn.relu,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer,
                name='dense_de'),
            tf.keras.layers.Reshape(target_shape=(int(sq_dim/2), n_conv)),
            tf.keras.layers.Conv1DTranspose(
                filters=n_conv, kernel_size=n_kernel, strides=2, padding='same', activation='relu',
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer,
                name='conv1dtrs_de'),
            tf.keras.layers.Conv1DTranspose(
                filters=1, kernel_size=n_kernel, strides=1, padding='same'),
            tf.keras.layers.Reshape((sq_dim,))
        ]
        )

latent_dim = 3
q_rs_dim = len(qq)-2

# Decoder layers
model_VAE = VAE(latent_dim, q_rs_dim)
model_decoder = model_VAE.decoder
model_decoder.summary()

# Augmented layers
# Add a dense layer to pre-trained decoder
regularizer = tf.keras.regularizers.L2(1.0)
dense_3 = [
    tf.keras.layers.InputLayer(input_shape=(3)),
    tf.keras.layers.Dense(6,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer,
                name='dense_in'),
    tf.keras.layers.Dense(6,
                kernel_regularizer = regularizer,
                bias_regularizer = regularizer,
                name='dense_in2'),]

# # rescaling the dense layer to the value range of decoder input
# rescale = [tf.keras.layers.Rescaling(scale=1.0, offset=0.0)]
# rescale[0].trainable = False

model_aug = tf.keras.Sequential(dense_3)#+rescale)
model_aug.summary()

## augmented decoder model
class Decoder_aug(tf.keras.Model):
    def __init__(self, latent_dim, sq_dim):
        super(Decoder_aug,self).__init__()
        self.latent_dim = latent_dim
        self.aug_layers = model_aug
        self.decoder_layers = model_decoder

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(64, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.aug_layers(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder_layers(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def sample_normal(self, x):
        mean, logvar = self.encode(x)
        eps = tf.random.normal(shape=(64, self.latent_dim))
        def zsample(e):
            return e*tf.exp(logvar*.5) + mean
        z_samples = tf.map_fn(zsample,eps)
        def logitsample(z):
            return self.decode(z, apply_sigmoid=True)
        logits_samples = tf.map_fn(logitsample,z_samples)

        logits_std = tf.math.reduce_std(logits_samples,axis=0)
        logits_mean = tf.math.reduce_mean(logits_samples,axis=0)

        return logits_mean, logits_std

model = Decoder_aug(latent_dim=3,sq_dim=q_rs_dim)

## load weights
export_path_aug = './saved_model/test/SQ_Gen_NN/'
model_name_aug = 'test_bg_model_aug_ft1'
export_name_aug = export_path_aug + model_name_aug

export_path_decoder = './saved_model/test/SQ_Gen_NN/'
model_name_decoder = 'test_bg_model_decoder_ft1'
export_name_decoder = export_path_decoder + model_name_decoder

aug_layers_loaded = model.aug_layers.load_weights(export_name_aug, by_name=False, skip_mismatch=False, options=None)
model.aug_layers = aug_layers_loaded._root

decoder_layers_loaded = model.decoder_layers.load_weights(export_name_decoder, by_name=False, skip_mismatch=False, options=None)
model.decoder_layers = decoder_layers_loaded._root

def IQ_gen_NN(parameters_in, GP=False, lmbda=0.5, sigma=0.1):
    parameters = [parameters_in[0],np.log(parameters_in[1]),parameters_in[2]]

    # mean and std of the training set labels
    mean = np.array([k_z_mean,kappa_mean,alpha_mean])
    std = np.array([k_z_std,kappa_std,alpha_std])
    parameters_z = [(parameters[i]-mean[i])/std[i] for i in range(3)]
    
    x = tf.reshape(to_tf(parameters_z),(1,3))
    mean, logvar = model.encode(x)
    # eps = model.reparameterize(mean, logvar)
    logits_mean = model.decode(mean, apply_sigmoid=True)
    predictions = f_out(logits_mean)

    return np.exp(predictions).flatten().astype('float64')

    if not GP:
        return np.exp(predictions).flatten().astype('float64')
    else:
        sample_mean_GP = sm_GP(predictions-np.mean(predictions),lmbda,sigma)+np.mean(predictions)
        return np.exp(sample_mean_GP).flatten().astype('float64')

with tf.device("/CPU:0"):
    def IQ_gen_NN_cpu(parameters_in, GP=False, lmbda=0.5, sigma=0.1):
        parameters = [parameters_in[0],np.log(parameters_in[1]),parameters_in[2]]

        # mean and std of the training set labels
        mean = np.array([k_z_mean,kappa_mean,alpha_mean])
        std = np.array([k_z_std,kappa_std,alpha_std])
        parameters_z = [(parameters[i]-mean[i])/std[i] for i in range(3)]
        
        x = tf.reshape(to_tf(parameters_z),(1,3))
        mean, logvar = model.encode(x)
        # eps = model.reparameterize(mean, logvar)
        logits_mean = model.decode(mean, apply_sigmoid=True)
        predictions = f_out(logits_mean)

        return np.exp(predictions).flatten().astype('float64')

        if not GP:
            return np.exp(predictions).flatten().astype('float64')
        else:
            sample_mean_GP = sm_GP(predictions-np.mean(predictions),lmbda,sigma)+np.mean(predictions)
            return np.exp(sample_mean_GP).flatten().astype('float64')