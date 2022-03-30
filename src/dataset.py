import os
import time 
import glob

import yogadl
import yogadl.storage

import numpy as np
import tensorflow as tf

from functools import partial
from skimage import io, color

from src.operators.NUFFT2D import NUFFT2D
from src.operators.NUFFT2D_TF import NUFFT2D_TF

class Dataset(tf.data.Dataset):

    @staticmethod
    def _generator(num_samples, files=[]):
        """yields the next image from the list of images `files`"""
        sample_idx = 0
        while True:
            filename = str(files[sample_idx])
            if filename.startswith("b\'"): # filter out some weird behaviour with byte strings
                filename = filename[2:-1]
            im = io.imread(filename, as_gray=True)
            if type(im[0,0]) == np.uint8: # some images are in uint for some reason
                im = im/ 255.
            im = im[:,:, np.newaxis] # reshaping to (None, None, 1)

            yield im
            sample_idx = (sample_idx + 1) % len(files) 

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
    """generates an image with random ellipses (between `nlow`  and `nhigh` ellipses)"""
    Nd = 256
    n = np.random.randint(nlow, nhigh)
    y,x = np.mgrid[:Nd,:Nd]
    a = np.random.uniform(0.1, 0.3, size = n).reshape(-1,1,1)*Nd
    b = np.random.uniform(0.1, 0.3, size = n).reshape(-1,1,1) *Nd

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
    """tensorflow dataset that yields images with random ellipses"""
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
    """Creates a dataset which can shuffle much faster than tf.dataset.shuffle, 
    if it doesn't work it returns a regular tf dataset"""
    try:
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
    except:
        print("Using YogaDL shuffle dataset failed, using tf shuffle instead (this will be significantly slower)")
        return tf_dataset.shuffle(len(tf_dataset), seed=42, reshuffle_each_iteration=True)

def random_crop(x):
    """creates a randomly selected, randomly flipped crop of size (256,256)"""
    data = tf.image.random_crop(x, size=(256,256,1)) #TODO adapt this to be adaptable
    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)
    return data

def center_crop(x):
    """creates a randomly flipped crop of the center of the image with size (256,256)"""
    data = tf.squeeze(x)
    data = tf.image.resize_with_crop_or_pad(x, 256, 256)
    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)
    return data


def measurement(x, m_op, ISNR, data_shape, Nd=(256,256)):
    """Creates measurement from image, adds measurement noise,
    returns measurement, true image"""
    y0 = m_op.dir_op(x.reshape(1, Nd[0], Nd[1]))

    sigma = np.sqrt(np.mean(np.abs(y0)**2)) * 10**(-ISNR/20)
    n = np.random.normal( 0, sigma, data_shape) + 1j * np.random.normal( 0, sigma, data_shape)
    y = y0 + n

    y_dirty = y
    return y_dirty.reshape(data_shape).astype(np.complex64), x.reshape(Nd).astype(np.float32)


def measurement_func(uv, m_op = None, data_shape=None, Nd=(256,256), ISNR=30):
    """function for getting a tf function version of the measurment function"""
    if data_shape is None:
        data_shape = (len(uv),)

    # TODO handle this operator neatly
    if m_op is None:
        m_op = NUFFT2D()
        m_op.plan(uv, (Nd[0], Nd[1]), (Nd[0]*2, Nd[1]*2), (6,6))
        m_op_tf = NUFFT2D_TF()
        m_op_tf.plan(uv, (Nd[0], Nd[1]), (Nd[0]*2, Nd[1]*2), (6,6), 20)

    func = partial(measurement, m_op=m_op, ISNR=ISNR, data_shape=data_shape, Nd=Nd)

    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
    def tf_function(x):
        return tf.numpy_function(func, [x], (tf.complex64, tf.float32))

    return tf_function, func

@tf.function()
def data_map(y, z, y_size=4440, z_size=(256,256)):
    """sets the shape of the objects (necessary for batching)"""
    y.set_shape([None,y_size])
    z.set_shape([None,z_size[0],z_size[1]])
    return y, z

@tf.function()
def data_map_image(y,z, y_size=4440, z_size=(256,256)):
    """sets the shape of the objects, expects image-to-image net (necessary for batching)"""
    # TODO update this, doesn't need y_shape
    y.set_shape([None,z_size[0],z_size[1]])
    z.set_shape([None,z_size[0],z_size[1]])
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


class PregeneratedDataset(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, operator="NUFFT_SPIDER"):
        i = 0
        try:
            operator = operator.decode('utf-8')
        except:
            pass
        while True:
            x = np.load(f"./data/intermediate/COCO/{operator}/x_true_train_30dB_{i:03d}.npy")
            y = np.load(f"./data/intermediate/COCO/{operator}/y_dirty_train_30dB_{i:03d}.npy")

            yield y, x
            i = (i + 1) % 100 # only a 100 presaved so reuse them

    def __new__(cls, operator, epochs=100):
        # assert os.path.exists(
            # f"./data/intermediate/COCO/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.complex64, tf.float32),
            args=(epochs, operator)
        )