import tensorflow as tf
import time 
from src.operators.measurement import NUFFT_op, NUFFT_op_TF
from src.sampling.uv_sampling import spider_sampling
# from scipy.io import loadmat
from skimage import io, color
import numpy as np
import os
import glob
from functools import partial


class id_op():
    @staticmethod
    def dir_op(x):
        return x
    @staticmethod
    def adj_op(x):
        return x
    @staticmethod
    def self_adj(x):
        return x


class Dataset(tf.data.Dataset):
    def __init__(self, m_op=id_op(), ISNR=50, weights=1):
        super().__init__()
        self.ISNR = ISNR
        self.weights = 1

    @staticmethod
    def _generator(num_samples):
        # Opening the file
        files = glob.glob("./data/BSR/BSDS500/data/images/test/*.jpg")
        for sample_idx in range(num_samples):
            data = io.imread(files[sample_idx])[:256,:256]
            data = color.rgb2gray(data)
            # data = m_op.dir_op(data)
            data = data_augmentation(data)
            data = mapping_function(data)
            yield data

    def __new__(cls, num_samples=200):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.complex64,
            # output_signature = tf.TensorSpec(shape = data_shape, dtype = tf.complex64),
            args=(num_samples,)
        )

def data_augmentation(x):
    """rotating and mirroring images randomly """
    p = np.random.rand()
    k1 = np.random.rand()//0.25
    x = np.rot90(x, k=k1)

    k2 = np.random.rand()//0.50
    if k2 == 1:
        x = np.fliplr(x)
    return x


def mapping_function(x):
    y0 = m_op.dir_op(x)

    sigma = np.sqrt(np.mean(np.abs(y0)**2)) * 10**(-ISNR/20)
    n = np.random.normal( 0, sigma, data_shape) + 1j * np.random.normal( 0, sigma, data_shape)
    y = y0 + n
    return m_op.adj_op(y*w)

def benchmark(dataset, num_epochs=10):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            # time.sleep(0.01)
            pass
    print("Execution time:", time.perf_counter() - start_time)


uv = spider_sampling()
m_op = NUFFT_op(uv)
# m_op = id_op()
data_shape = m_op.dir_op(np.zeros((256,256), dtype=complex)).shape
print(data_shape)
ISNR = 50
dist = -1 * np.dot(uv, uv.T) -1 * np.dot(uv, uv.T).T + np.sum(uv**2, axis=1) + np.sum(uv**2, axis=1)[:,np.newaxis] # (x-y)'(x-y) = x'x + y'y - x'y -y'x
dist[dist < 0] = 0 # correct for numerical errors
gridsize = 2*np.pi/512
w = 1/np.sum(dist**.5 < gridsize, axis=1) # all pixels within 1 gridcell distance

# w = 1

ds = Dataset()
benchmark(
    ds.prefetch(tf.data.AUTOTUNE)
        )