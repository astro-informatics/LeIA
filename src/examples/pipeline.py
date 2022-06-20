from functools import partial
from tensorflow.python.framework.ops import disable_eager_execution

from src.operators.NUFFT2D import NUFFT2D
disable_eager_execution()

import os
import sys 
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

# operators and sampling patterns
from src.operators.NUFFT2D_TF import NUFFT2D_TF

from src.sampling.uv_sampling import spider_sampling

 # some custom callbacks
from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 

# model and dataset generator
from src.networks.UNet import UNet
from src.dataset import Dataset, PregeneratedDataset, data_map, make_yogadl_dataset, measurement_func, random_crop

# selecting one gpu to train on
from src.util import gpu_setup
gpu_setup()


class Pipeline():
    def __init__(self):
        pass

    def create_network(self):
        model = self.network(
            
        )

    def setup_dataset(self):
        """Tries to use pre-augmented data, otherwise creates a new dataset with augmentation"""
        try:  
            dataset = PregeneratedDataset(
            operator=operator, epochs=epochs
            ).unbatch().batch(
                batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE
            ).map(
                lambda x, y: data_map(x,y, y_size=len(uv)), 
                num_parallel_calls=tf.data.AUTOTUNE
                ).prefetch(set_size//batch_size)
        except:
            tf_func, _ = measurement_func(uv,  m_op=op, Nd=(256,256), data_shape=y_shape, ISNR=ISNR)
            ds = Dataset(set_size, data)
            yogadl_dataset = make_yogadl_dataset(ds) # use yogadl for caching and shuffling the data
            data_map = partial(data_map, y_size=y_shape, z_size=(256,256))
            dataset = yogadl_dataset.map(random_crop).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)
    
        self.dataset = dataset

    def train_model(self):
        self.network.fit(self.dataset)

    def run_all(self):
        self.create_network()
        self.setup_dataset()
        self.train_model()
        self.make_predictions()
        self.calculate_statistics()