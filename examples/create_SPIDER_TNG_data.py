# script to create some pre-augmented datasets for training and evaluation purposes
from src.util import gpu_setup
gpu_setup()

import os
import sys
import yaml

import numpy as np
import tensorflow as tf

from src.data.SPIDER_datasets import TNGDataset
from src.dataset import measurement_func, random_crop
from src.sampling.uv_sampling import spider_sampling

config_file = str(sys.argv[1])
with open(config_file, "r") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

pre_augment_epochs = 100 # prepare 100 different pre-augmented sets
train_size = cfg.train_size
test_size = cfg.test_size

data_folder = f"{cfg.data_base_folder}/intermediate/{cfg.dataset}/{cfg.operator}/"
os.makedirs(data_folder, exist_ok=True)

Nd = (cfg.Nd, cfg.Nd)
Kd = (cfg.Kd, cfg.Kd)
Jd = (cfg.Jd,cfg.Jd)
ISNR = cfg.ISNR #dB

batch_size = 1

uv_original = spider_sampling()

# function on true uv coverage
tf_func_original, func_original = measurement_func(uv_original,  m_op=None, Nd=Nd, ISNR=ISNR)

# set random seeds
np.random.seed(8394829)
tf.random.set_seed(8394829)

ds = TNGDataset().unbatch().map(tf_func_original).take(484)
array = list(ds.as_numpy_iterator())
y_data = np.array([x[0] for x in array])
x_data = np.array([x[1] for x in array])

x_train = x_data[:train_size].reshape(-1, Nd[0], Nd[1])
y_train = y_data[:train_size].reshape(-1, len(uv_original))
x_test = x_data[train_size:len(x_data)-len(x_data)%batch_size].reshape(-1, Nd[0], Nd[1])
y_test = y_data[train_size:len(x_data)-len(x_data)%batch_size].reshape(-1, len(uv_original))

folder = f"./data/intermediate/TNG/NUFFT_SPIDER/"
os.makedirs(folder, exist_ok=True)

np.save(f"{folder}/x_true_train_{ISNR}dB.npy",  x_train)
np.save(f"{folder}/y_dirty_train_{ISNR}dB.npy", y_train)
np.save(f"{folder}/x_true_test_{ISNR}dB.npy",   x_test)
np.save(f"{folder}/y_dirty_test_{ISNR}dB.npy",  y_test)