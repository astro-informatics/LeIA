import tensorflow as tf
import time 
from src.operators.measurement import NUFFT_op
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
#         files = glob.glob(os.environ["HOME"] + "/src_aiai/data/val2017/*.jpg")
        files = np.loadtxt(os.environ["HOME"] + 
            "/src_aiai/images.txt", dtype=str) # only a selection of the files are larger than (256,256)
        for sample_idx in range(num_samples):
            if sample_idx >= len(files):
                print(sample_idx)
            
            im = io.imread(files[sample_idx], as_gray=True)
            if type(im[0,0]) == np.uint8: # some images are in uint for some reason
                im = im/ 255.
            im = im[:,:, np.newaxis] # reshaping to (None, None, 1)
            
#             data = io.imread(files[sample_idx])
#             data = color.rgb2gray(data)
#             data = data[:,:, np.newaxis]
#             data = tf.convert_to_tensor(data)
#             data = tf.io.read_file(files[sample_idx])
#             data = tf.io.decode_jpeg(data, channels=1)

            yield im

    def __new__(cls, num_samples=200):
        return tf.data.Dataset.from_generator(
            cls._generator,
            # output_types=(tf.float32, tf.float32),
            output_types= tf.float32,
#             output_shapes=(tf.TensorShape([256,256,1]),),
#             output_signature = tf.TensorSpec(shape = (256,256,1), dtype = tf.uint8),
            args=(num_samples,)
        )


def crop(x):
#     x = tf.cast(x, tf.float32)/255
    data = tf.image.random_crop(x, size=(256,256,1))
    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)
    return data



def measurement(x, m_op, ISNR, data_shape, w):
    y0 = m_op.dir_op(x.reshape(256,256))

    sigma = np.sqrt(np.mean(np.abs(y0)**2)) * 10**(-ISNR/20)
    n = np.random.normal( 0, sigma, data_shape) + 1j * np.random.normal( 0, sigma, data_shape)
    y = y0 + n
    
#     x_dirty = m_op.adj_op(y*w).real
#     fft = lambda x: tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(x, axes=(-2,-1))), axes=(-2,-1))
    # fft = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x), norm='ortho'))
    
    y_dirty = y
#     return ((x_dirty.reshape(256,256,1), y.reshape(4440,1)), x.reshape(256,256,1))
#     print(x_dirty.shape, y_dirty.shape)
#     return (x_dirty.reshape(256,256,1).astype(np.float32), y_dirty.reshape(4440,1).astype(np.complex128), x.reshape(256,256,1).astype(np.float32))
    return y_dirty.reshape(4440).astype(np.complex64), x.reshape(256,256).astype(np.float32))

#     return (x_dirty, y_dirty, x)

# @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
# def tf_function(x):
#     return tf.numpy_function(mapping_function, [x], (tf.float32, tf.float32))
#     # return tf.py_function(mapping_function, [x], (tf.float32, tf.float32))

def measurement_func(ISNR=50, data_shape=(4440,)):
    """function for getting a tf function version of the measurment function"""
    uv = spider_sampling()
    m_op = NUFFT_op()
    m_op.plan(uv, (256,256), (512,512), (6,6))

    grid_cell = 2*np.pi /512 
    binned = (uv[:,:]+np.pi+.5*grid_cell) // grid_cell
    binned = [tuple(x) for x in binned]
    cells = set(binned)
    w_gridded = np.zeros(uv.shape[0])
    for cell in list(cells):
        mask = np.all(np.array(cell) ==  binned, axis=1)
        w_gridded[mask] = np.sum(mask)

    # w = 
    w = 1/w_gridded
    w /= w.max()

    func = partial(measurement, m_op=m_op, ISNR=ISNR, data_shape=data_shape, w=w)

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def tf_function(x):
#         return tf.numpy_function(func, [x], ((tf.float32, tf.complex128), tf.float32))
#         return tf.numpy_function(func, [x], (tf.float32, tf.complex128, tf.float32))
        return tf.numpy_function(func, [x], (tf.complex64, tf.float32))

    return tf_function, func

@tf.function()
def data_map(y,z):
    """split input and output of train data"""
#     return {"input_1":x, "input_3":y}, z
#     x = tf.expand_dims(x, 3)
#     x.set_shape([None,256,256,1])
#     y = tf.expand_dims(y, 3)
    y.set_shape([None,4440])
#     z = tf.expand_dims(z, 3)
    z.set_shape([None,256,256])
#     return (x, y), z
    return y, z


def benchmark(dataset, num_epochs=10):
    start_time = time.perf_counter()
    last_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            pass
        print(f"Execution time {epoch_num}:", time.perf_counter() - last_time)
        last_time = time.perf_counter()
                
    print("Execution time:", time.perf_counter() - start_time)

def set_shape(x):
    a, b = x
    a.set_shape([256,256,1])
    b.set_shape([256,256,1])
    return a, b
