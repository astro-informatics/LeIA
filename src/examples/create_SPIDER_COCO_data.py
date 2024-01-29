# script to create some pre-augmented datasets for training and evaluation purposes
from src.util import gpu_setup
gpu_setup()

import os
import sys
import yaml
import tqdm

import numpy as np
import tensorflow as tf

from src.data.SPIDER_datasets import COCODataset
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

ds = COCODataset(train_size + test_size, "COCO")

print("start creating datasets")
for i in tqdm.tqdm(range(pre_augment_epochs+1)):

    if i == 0:
        dataset = ds.take(train_size + test_size).map(tf_func_original)
        array = list(dataset.as_numpy_iterator())
        y_data = np.array([x[0] for x in array])
        x_data = np.array([x[1] for x in array])

        np.save(f"{data_folder}/x_true_train_{ISNR}dB.npy",  x_data[:train_size])
        np.save(f"{data_folder}/x_true_test_{ISNR}dB.npy",   x_data[train_size:])
        np.save(f"{data_folder}/y_dirty_train_{ISNR}dB.npy", y_data[:train_size])
        np.save(f"{data_folder}/y_dirty_test_{ISNR}dB.npy",  y_data[train_size:])
        dataset = ds.take(train_size).cache()
    else:
        dataset2 = dataset.shuffle(train_size).map(random_crop).map(tf_func_original)

        array = list(dataset2.as_numpy_iterator())

        y_data = np.array([x[0] for x in array])
        x_data = np.array([x[1] for x in array])      

        np.save(f"{data_folder}/x_true_train_{ISNR}dB_{i-1:03d}.npy",  x_data)
        np.save(f"{data_folder}/y_dirty_train_{ISNR}dB_{i-1:03d}.npy", y_data)


# Create set with varying ISNR
x_robustness = []
y_robustness = []
dataset = ds.take(train_size+100).cache()

for isnr in tqdm.tqdm(np.arange(30, 5,-2.5)):
    tf_func, func = measurement_func(uv_original,  m_op=None, Nd=Nd, ISNR=isnr)

    dataset2 = dataset.skip(train_size).take(100).map(random_crop).map(tf_func)
    array = list(dataset2.as_numpy_iterator())

    y_data = np.array([x[0] for x in array])
    x_data = np.array([x[1] for x in array])

    y_robustness.append(y_data)
    x_robustness.append(x_data)

np.save(f"{data_folder}/x_true_test_{ISNR}dB_robustness.npy",  np.array(x_robustness).reshape(-1, 256,256))
np.save(f"{data_folder}/y_dirty_test_{ISNR}dB_robustness.npy",  np.array(y_robustness).reshape(-1, len(uv_original)))