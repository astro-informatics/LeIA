from src.util import gpu_setup
gpu_setup()

import os
import sys
import time
import yaml

import numpy as np
import tensorflow as tf
from astropy.io import fits

from src.networks.UNet import UNet
#from src.networks.GUNet_var import GUNet_var
from src.networks.GUNet_var2 import GUNet_var

from src.operators.NUFFT2D_TF import NUFFT2D_TF

from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 
from src.util import PSNRMetric
from src.data.RAI_datasets import Known, Fixed, Validation, Varying

config_file = str(sys.argv[1])
with open(config_file, "r") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

### load values from config file
operator = cfg.get("operator", "NUFFT_Random_var")
dataset = cfg.get("dataset", "TNG")
network = cfg.get("network", "UNet")
ISNR = cfg.get("ISNR", 30)

# mode = cfg.get("training_strategy", "True")
exp_name = cfg.get("exp_name", "test")

data_base_folder = cfg.get("data_base_folder", "./data/")
checkpoint_base_folder = cfg.get("checkpoint_base_folder", "./models/")
log_base_folder = cfg.get("log_base_folder", "./logs/")
results_base_folder = cfg.get("results_base_folder", "./results/")

Nd = cfg.get("Nd", 256) # input image size
Kd = cfg.get("Kd", 512) # upsampled image size (for use in NUFFT operator)
Jd = cfg.get("Jd", 6)   # NUFFT kernel size

epochs = cfg.get("epochs", 100)
transfer_epochs = cfg.get("transfer_epochs", 100)

batch_size = cfg.get("batch_size", 2)
train_size = cfg.get("train_size", 2000)
val_size = cfg.get("val_size", 1000)
test_size = cfg.get("test_size", 1000)
save_freq = cfg.get("save_freq", 5)

batch_size = 4

postfix = f"_{exp_name}"

data_folder = f"{data_base_folder}/intermediate/{dataset}/{operator}/"
checkpoint_folder = f"{checkpoint_base_folder}/{dataset}/{operator}/{network}_{ISNR}dB_{exp_name}"
checkpoint_path = checkpoint_folder + "/cp-{epoch:04d}.ckpt"


# if mode == "True":
#     uv = np.load(data_folder + "/uv_original.npy")
# else:
# uv = np.load(data_folder + "/uv_big.npy")

d =  fits.getdata("./notebooks/my.uvfits")
u, v, w = d['UU'], d['VV'], d['WW']
uv_radio = np.vstack((d['UU'], d['VV'])).T
uv_radio = uv_radio /np.linalg.norm(uv_radio, axis=1).max()*np.pi
uv_radio = uv_radio

def calc_w(uv):
    grid_cell = 2*np.pi /512 
    binned = (uv[:,:]+np.pi+.5*grid_cell) // grid_cell
    binned = [tuple(x) for x in binned]
    w_gridded = np.zeros(uv.shape[0])

    d = {}
    for i in binned:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1

    for i in range(len(w_gridded)):
        w_gridded[i] = d[binned[i]]
    # w = 
    w = 1/w_gridded
    w /= w.max()
    return w

w_radio = calc_w(uv_radio)

gunet = GUNet_var(
    (256,256), 
    uv=uv_radio,
    op=NUFFT2D_TF, 
    depth=4, 
    conv_layers=2,
    input_type="measurements", 
    measurement_weights=w_radio,
    batch_size=batch_size,
    residual=True,
    metrics=[PSNRMetric()]
    )

# operator = ""
# network = "GUNet_var"
# postfix = "_specific_new2"

checkpoint_folder = "./models/TNG/NUFFT_Random_var/GUNet_var_30dB_specific_new2"
latest = tf.train.latest_checkpoint(checkpoint_folder)
gunet.load_weights(latest)

x_train = np.load('data/intermediate/TNG_meerkat/NUFFT_Random_var/x_true_train_30dB.npy')
y_train = np.load('data/intermediate/TNG_meerkat/NUFFT_Random_var/y_dirty_train_30dB.npy')

x_test = np.load('data/intermediate/TNG_meerkat/NUFFT_Random_var/x_true_test_30dB.npy')
y_test = np.load('data/intermediate/TNG_meerkat/NUFFT_Random_var/y_dirty_test_30dB.npy')

z_train = np.ones_like(y_train)
z_test = np.ones_like(y_test)

### Callbacks
csv_logger = CSV_logger_plus(f"{log_base_folder}/{dataset}/{operator}/log_{network}_{ISNR}dB" + postfix + "2")

checkpoint_path = "./models/TNG/NUFFT_Random_var/GUNet_var_30dB_meerkat_no_weighting2/GUNet_var_30dB_meerkat_no_weighting3"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    monitor='val_PSNR',
    mode='max',
    save_best_only='True',
    save_weights_only=True,
    save_freq='epoch'# save_freq* (train_size//batch_size)
)

epochs = 100
gunet.fit((y_train, z_train), x_train, validation_data=((y_test, z_test), x_test), epochs = epochs, batch_size=batch_size, callbacks=[csv_logger, cp_callback])

pred_train = gunet.predict((y_train, z_train), batch_size=batch_size)
pred_test = gunet.predict((y_test, z_test), batch_size=batch_size)

np.save("/share/gpu0/mars/TNG_data/meerkat_clean/pred/pred_train_GUNet_full_transfer_no_weighting2.npy", pred_train)
np.save("/share/gpu0/mars/TNG_data/meerkat_clean/pred/pred_test_GUNet_full_transfer_no_weighting2.npy", pred_test)

