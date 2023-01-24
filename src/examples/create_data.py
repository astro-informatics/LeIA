 # script to create some pre-augmented datasets for training and evaluation purposes
from src.util import gpu_setup
gpu_setup()

import os
import sys
import tqdm
import numpy as np
import tensorflow as tf
import glob

from src.dataset import measurement_func, Dataset, random_crop, center_crop, EllipseDataset
from src.operators.NUFFT2D import NUFFT2D
from src.operators.NNFFT2D import NNFFT2D
from src.sampling.uv_sampling import spider_sampling, random_sampling



Nd = (256, 256)
ISNR = 30 #dB

epochs = 100
train_size = 2000
test_size = 1000
n_operators = 100

# data = 'COCO'
data = 'TNG'
random = True
operator = sys.argv[1]
project_folder = os.environ["HOME"] +"/src_aiai/"


if random:
    operator += "_var"


folder = project_folder + f"data/intermediate/{data}/{operator}/"
os.makedirs(folder, exist_ok=True)

Nd = (256, 256)
Kd = (512, 512)
Jd = (6,6)

batch_size = 1
if operator == "NUFFT_SPIDER":
    uv = spider_sampling()
    uv_original = uv
    y_shape = len(uv)
    m_op = NUFFT2D()
    m_op.plan(uv, Nd, Kd, Jd, batch_size=batch_size)
    m_op_original = m_op
elif operator == "NUFFT_Random" or operator == "NUFFT_Random_var":
    y_shape = int(Nd[0]**2/2)
    try: 
        uv_original = np.load(f"{folder}/uv_original.npy")
    except:
        uv_original = random_sampling(y_shape)
        np.save(f"{folder}/uv_original.npy", uv_original)
    if random:
        y_shape *=2
    uv = random_sampling(y_shape)
    if random:
        try:
            uv = np.load(f"{folder}/uv_big.npy", uv)
        except:
            np.save(f"{folder}/uv_big.npy", uv)
    
#     m_op_original = NUFFT2D()
#     m_op_original.plan(uv_original, Nd, Kd, Jd, batch_size=batch_size)
elif operator == "NNFFT_Random":
    y_shape = int(Nd[0]**2/2)
    uv = random_sampling(y_shape)
    m_op = NNFFT2D()
    m_op.plan(uv, Nd, Kd, Jd, batch_size=batch_size)

tf_func_original, func_original = measurement_func(uv_original,  m_op=None, Nd=Nd, ISNR=ISNR)
tf_func = tf_func_original


if random:
    print("generating random operator sub-sets")
    tf_func, func_big = measurement_func(uv,  m_op=None, Nd=Nd, ISNR=ISNR)



    
    
np.random.seed(8394829)
tf.random.set_seed(8394829)

class TNGDataset(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

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

if data == "COCO":
    ds = Dataset(train_size + test_size, data)
elif data == "TNG":
    ds = TNGDataset().unbatch()


print("start creating datasets")
for i in tqdm.tqdm(range(epochs+1)):


    if i == 0:
        dataset = ds.take(train_size + test_size).map(random_crop).map(tf_func_original)
        array = list(dataset.as_numpy_iterator())
        y_data = np.array([x[0] for x in array])
        x_data = np.array([x[1] for x in array])

        np.save(f"{folder}/x_true_train_{ISNR}dB.npy",  x_data[:train_size])
        np.save(f"{folder}/x_true_test_{ISNR}dB.npy",   x_data[train_size:])
        np.save(f"{folder}/y_dirty_train_{ISNR}dB.npy", y_data[:train_size])
        np.save(f"{folder}/y_dirty_test_{ISNR}dB.npy",  y_data[train_size:])
        dataset = ds.take(train_size).cache()
    else:
        # if random:
            # sel_data = []
            # for idx in range(train_size):
            #     if idx % 20 == 0:
            #         batch_sel = np.random.permutation(len(uv)) > len(uv)/2

            #     sel_data.append(batch_sel)                         
            # np.save(f"{folder}/sel_train_{ISNR}dB_{i-1:03d}.npy", sel_data) 

        dataset2 = dataset.shuffle(train_size).map(random_crop).map(tf_func)
        array = list(dataset2.as_numpy_iterator())

        y_data = np.array([x[0] for x in array])
        x_data = np.array([x[1] for x in array])      

        np.save(f"{folder}/x_true_train_{ISNR}dB_{i-1:03d}.npy",  x_data)
        np.save(f"{folder}/y_dirty_train_{ISNR}dB_{i-1:03d}.npy", y_data)


exit()

# set with varying ISNR
x_robustness = []
y_robustness = []
dataset = ds.take(train_size+100).cache()
folder = project_folder + f"data/intermediate/{data}/{operator}/"

for isnr in tqdm.tqdm(np.arange(30, 5,-2.5)):
    tf_func, func = measurement_func(uv_original,  m_op=m_op_original, Nd=Nd, ISNR=isnr)

    dataset2 = dataset.skip(set_size).take(100).map(random_crop).map(tf_func)
    array = list(dataset2.as_numpy_iterator())

    y_data = np.array([x[0] for x in array])
    x_data = np.array([x[1] for x in array])

    y_robustness.append(y_data)
    x_robustness.append(x_data)

np.save(f"{folder}/x_true_test_{ISNR}dB_robustness.npy",  np.array(x_robustness).reshape(-1, 256,256))
np.save(f"{folder}/y_dirty_test_{ISNR}dB_robustness.npy",  np.array(y_robustness).reshape(-1, m_op_original.n_measurements))
