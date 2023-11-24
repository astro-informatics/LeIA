 # script to create some pre-augmented datasets for training and evaluation purposes
from src.util import gpu_setup
gpu_setup()

import os
import sys
import yaml
import glob
import tqdm

import numpy as np
import tensorflow as tf

from src.dataset import measurement_func
from src.sampling.uv_sampling import random_sampling

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

try: 
    uv = np.load(f"{data_folder}/uv_big.npy")
    sel_original = np.load(f"{data_folder}/sel.npy")
    uv_original = np.load(f"{data_folder}/uv_original.npy")
except:
    y_shape = int(Nd[0]**2/2)
    uv = random_sampling(y_shape*2)
    sel_original = np.random.permutation(len(uv)) < len(uv)/2
    uv_original = uv[sel_original]
    np.save(f"{data_folder}/uv_big.npy", uv)
    np.save(f"{data_folder}/sel.npy", sel_original)
    np.save(f"{data_folder}/uv_original.npy", uv_original)

# function on true uv coverage
tf_func_original, func_original = measurement_func(uv_original,  m_op=None, Nd=Nd, ISNR=ISNR)
# function on oversampled uv coverage
tf_func, func_big = measurement_func(uv,  m_op=None, Nd=Nd, ISNR=ISNR)

# set random seeds
np.random.seed(8394829)
tf.random.set_seed(8394829)

class TNGDataset(tf.data.Dataset):
    """a dataset that loads image data. """

    @staticmethod
    def _generator():
        files = glob.glob("/home/mars/git/IllustrisTNG/data/processed_256/TNG*.npy")
        x = np.array([np.load(file) for file in files])
        while True:
            yield x[:,:,:,np.newaxis]

    def __new__(cls):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float32),
            args=()
        )

ds = TNGDataset().unbatch()

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
        dataset2 = dataset.shuffle(train_size).map(tf_func)
        array = list(dataset2.as_numpy_iterator())

        y_data = np.array([x[0] for x in array])
        x_data = np.array([x[1] for x in array])      

        np.save(f"{data_folder}/x_true_train_{ISNR}dB_{i-1:03d}.npy",  x_data)
        np.save(f"{data_folder}/y_dirty_train_{ISNR}dB_{i-1:03d}.npy", y_data)