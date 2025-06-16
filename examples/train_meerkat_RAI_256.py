import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from src.util import gpu_setup
gpu_setup(40000)
import tensorflow as tf

from src.networks.GUNet_PSF import GUNet
from src.util import PSNRMetric
from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 

data_base_folder= "./data/"
checkpoint_base_folder= "./models/"
log_base_folder= "./logs/"
results_base_folder= "./results/"
operator = "NUFFT_Random_var"
network = "GUNet_PSF"
ISNR = 30
postfix = ""
dataset = "TNG"
mode = "psf_var"
exp_name = "meerkat_256_transfer"

checkpoint_path = f"{checkpoint_base_folder}/{dataset}/{operator}/{network}_{ISNR}dB_{mode}_{exp_name}"

x_train =  np.load("./data/intermediate/TNG_meerkat/NUFFT_Random_var/x_true_train_30dB.npy")
y_train =  np.load("./data/intermediate/TNG_meerkat/NUFFT_Random_var/train_dirty.npy")
x0_train = np.load("./data/intermediate/TNG_meerkat/NUFFT_Random_var/train_x0.npy")
uv_train =  np.load("./data/intermediate/TNG_meerkat/NUFFT_Random_var/train_psf.npy")



x_test =  np.load("./data/intermediate/TNG_meerkat/NUFFT_Random_var/x_true_test_30dB.npy")
y_test =  np.load("./data/intermediate/TNG_meerkat/NUFFT_Random_var/test_dirty.npy")
uv_test =  np.load("./data/intermediate/TNG_meerkat/NUFFT_Random_var/test_psf.npy")
x0_test = np.load("./data/intermediate/TNG_meerkat/NUFFT_Random_var/test_x0.npy")


batch_size = 20
Nd = 256
epochs = 50

model = GUNet(
    (Nd, Nd), 
    depth=4, 
    conv_layers=2,
    batch_size=batch_size,
    residual=False,
    metrics=[PSNRMetric()]
)

latest = tf.train.latest_checkpoint("./models/TNG/NUFFT_Random_var/GUNet_var_30dB_specific_known_new2/")
model.load_weights(latest)
csv_logger = CSV_logger_plus(f"./logs/TNG/NUFFT_Random_var/meerkat_30dB_no_weighting")


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    monitor='PSNR',
    mode='max',
    save_best_only='True',
    save_weights_only=True,
    save_freq='epoch'#save_freq* (train_size//batch_size)
)

model.fit((x0_train, y_train, uv_train), x_train, batch_size=batch_size, callbacks=[csv_logger, cp_callback], epochs=epochs)

pred_train = model.predict((x0_train, y_train, uv_train), batch_size=batch_size)
pred_test = model.predict((x0_test, y_test, uv_test), batch_size=batch_size)

# np.save("/share/gpu0/mars/TNG_data/meerkat_clean/pred/pred_train_GUNet.npy", pred_train)
# np.save("/share/gpu0/mars/TNG_data/meerkat_clean/pred/pred_test_GUNet.npy", pred_test)
np.save("/share/gpu0/mars/TNG_data/meerkat_clean/pred/pred_train_GUNet_psf_transfer.npy", pred_train)
np.save("/share/gpu0/mars/TNG_data/meerkat_clean/pred/pred_test_GUNet_psf_transfer.npy", pred_test)
