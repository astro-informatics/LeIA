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
import yogadl
import yogadl.storage

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

    @staticmethod
    def _generator(num_samples, files=[]):

        for sample_idx in range(num_samples):
            if sample_idx >= len(files):
                print(sample_idx)
            
            filename = str(files[sample_idx])
            if filename.startswith("b\'"): # filter out some weird behaviour with byte strings
                filename = filename[2:-1]
            im = io.imread(filename, as_gray=True)
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

    def __new__(cls, num_samples=200, data="COCO"):

        if data == "COCO":
            files = np.loadtxt(os.environ["HOME"] + 
                            "/src_aiai/images.txt", dtype=str) # only a selection of the files are larger than (256,256)    
        elif data == "GZOO":
            files = glob.glob(os.environ["HOME"] +"/src_aiai/data/galaxy_zoo_train/images_training_rev1/*.jpg")     
        elif data == "SATS":
            files = glob.glob(os.environ["HOME"] +"/src_aiai/data/sat/train/*.jpg")     
        else: 
            files = [] # this will not work but catches edge cases
        print(f"using dataset {data} with {len(files)} files")
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types= tf.float32,
            args=(num_samples, files)
        )


def random_ellipses(nlow, nhigh):
    Nd = 256
    n = np.random.randint(nlow, nhigh)
    y,x = np.mgrid[:Nd,:Nd]
    a = np.random.uniform(0.1, 0.3, size = n).reshape(-1,1,1)*Nd
    b = np.random.uniform(0.1, 0.3, size = n).reshape(-1,1,1) *Nd
    # x0 = np.random.uniform(0.2, 0.8, size=n).reshape(-1,1,1) *Nd
    # y0 = np.random.uniform(0.2, 0.8, size=n).reshape(-1,1,1) *Nd

    r0 = np.random.uniform(0, 0.3, size=n).reshape(-1,1,1) *Nd
    phi =  np.random.uniform(0, 2*np.pi, size=n).reshape(-1,1,1)
    
    x0 = r0*np.cos(phi) + Nd/2
    y0 = r0*np.sin(phi) + Nd/2
    
    p1 = x - x0
    p2 = y - y0

    theta =  np.random.uniform(0, np.pi, size=n).reshape(-1,1,1) # ellpse rotation
    amps = np.random.uniform(0.1, 1, size=n).reshape(-1,1,1)  # amplitude

    ell = np.sum( amps* ((p1*np.cos(phi) + p2*np.sin(phi))**2/a**2 + (p1*np.sin(phi) - p2*np.cos(phi))**2/b**2 < 1), axis=0) 
    return ell /ell.max()

class EllipseDataset(tf.data.Dataset):

    @staticmethod
    def _generator(num_samples):
        for i in range(num_samples):
            im = random_ellipses(7, 10)[:,:,np.newaxis]
            yield im

    def __new__(cls, num_samples=200):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types= tf.float32,
            args=(num_samples,)
        )


def make_yogadl_dataset(tf_dataset, storage_path="/tmp/yogadl_cache", shuffle=True):
    """
    Creates a dataset which can shuffle much faster than tf.dataset.shuffle"""
    os.makedirs(storage_path, exist_ok=True)
    lfs_config = yogadl.storage.LFSConfigurations(storage_path)
    storage = yogadl.storage.LFSStorage(lfs_config)

    @storage.cacheable('coco', '1')
    def make_data(dataset):
        return dataset

    # Get the DataRef from the storage via the decorated function.
    dataref = make_data(tf_dataset)
    
    stream = dataref.stream(
        shuffle=shuffle,
        shuffle_seed=42,
    )
    return yogadl.tensorflow.make_tf_dataset(stream)

def random_crop(x):
#     x = tf.cast(x, tf.float32)/255
    data = tf.image.random_crop(x, size=(256,256,1)) #TODO adapt this to be adaptable
    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)
    return data

def center_crop(x):
#     x = tf.cast(x, tf.float32)/255
    data = tf.squeeze(x)
    data = tf.image.resize_with_crop_or_pad(x, 256, 256)
    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)
    return data


def measurement(x, m_op, ISNR, data_shape, w, Nd=(256,256)):
    y0 = m_op.dir_op(x.reshape(Nd))

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
    return y_dirty.reshape(-1).astype(np.complex64), x.reshape(Nd).astype(np.float32)

#     return (x_dirty, y_dirty, x)

# @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
# def tf_function(x):
#     return tf.numpy_function(mapping_function, [x], (tf.float32, tf.float32))
#     # return tf.py_function(mapping_function, [x], (tf.float32, tf.float32))

def measurement_func(uv, Nd=(256,256), ISNR=30):
    data_shape = (len(uv),)
    """function for getting a tf function version of the measurment function"""
    # uv = spider_sampling()
    m_op = NUFFT_op()
    m_op.plan(uv, (Nd[0], Nd[1]), (Nd[0]*2, Nd[1]*2), (6,6))

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

    func = partial(measurement, m_op=m_op, ISNR=ISNR, data_shape=data_shape, w=w, Nd=Nd)

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def tf_function(x):
#         return tf.numpy_function(func, [x], ((tf.float32, tf.complex128), tf.float32))
#         return tf.numpy_function(func, [x], (tf.float32, tf.complex128, tf.float32))
        return tf.numpy_function(func, [x], (tf.complex64, tf.float32))

    return tf_function, func

@tf.function()
def data_map(y,z, y_size=4440, z_size=(256,256)):
    """split input and output of train data"""
#     return {"input_1":x, "input_3":y}, z
#     x = tf.expand_dims(x, 3)
#     x.set_shape([None,256,256,1])
#     y = tf.expand_dims(y, 3)
    y.set_shape([None,y_size])
#     z = tf.expand_dims(z, 3)
    z.set_shape([None,z_size[0],z_size[1]])
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
