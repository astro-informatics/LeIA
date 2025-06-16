import os
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from src.util import gpu_setup
gpu_setup()
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
exp_name = "meerkat"

checkpoint_path = f"{checkpoint_base_folder}/{dataset}/{operator}/{network}_{ISNR}dB_{mode}_{exp_name}"

x_train =  np.load("/share/gpu0/mars/TNG_data/rcGAN/meerkat_clean/train/x.npy")[:,4:-4,4:-4]
y_train =  np.load("/share/gpu0/mars/TNG_data/rcGAN/meerkat_clean/train/y.npy")[:,4:-4,4:-4]
uv_train =  np.load("/share/gpu0/mars/TNG_data/rcGAN/meerkat_clean/train/uv.npy")[:,4:-4,4:-4]


x_val =  np.load("/share/gpu0/mars/TNG_data/rcGAN/meerkat_clean/val/x.npy")[:,4:-4,4:-4]
y_val =  np.load("/share/gpu0/mars/TNG_data/rcGAN/meerkat_clean/val/y.npy")[:,4:-4,4:-4]
uv_val =  np.load("/share/gpu0/mars/TNG_data/rcGAN/meerkat_clean/val/uv.npy")[:,4:-4,4:-4]

x_test =  np.load("/share/gpu0/mars/TNG_data/rcGAN/meerkat_clean/test/x.npy")[:,4:-4,4:-4]
y_test =  np.load("/share/gpu0/mars/TNG_data/rcGAN/meerkat_clean/test/y.npy")[:,4:-4,4:-4]
uv_test =  np.load("/share/gpu0/mars/TNG_data/rcGAN/meerkat_clean/test/uv.npy")[:,4:-4,4:-4]

batch_size = 20
Nd = 352
epochs = 100

model = GUNet(
    (Nd, Nd), 
    depth=4, 
    conv_layers=2,
    batch_size=batch_size,
    residual=False,
    metrics=[PSNRMetric()]
)


csv_logger = CSV_logger_plus(f"{log_base_folder}/{dataset}/{operator}/log_{network}_{ISNR}dB" + postfix + "")


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    monitor='val_PSNR',
    mode='max',
    save_best_only='True',
    save_weights_only=True,
    save_freq='epoch'#save_freq* (train_size//batch_size)
)

model.fit((y_train, uv_train), x_train, batch_size=batch_size, validation_data= ((y_val, uv_val), x_val), callbacks=[csv_logger, cp_callback], epochs=epochs)

pred_train = model.predict((y_train, uv_train), batch_size=batch_size)
pred_test = model.predict((y_test, uv_test), batch_size=batch_size)

np.save("/share/gpu0/mars/TNG_data/meerkat_clean/pred/pred_train_GUNet.npy", pred_train)
np.save("/share/gpu0/mars/TNG_data/meerkat_clean/pred/pred_test_GUNet.npy", pred_test)