#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import time


# In[3]:


from functools import partial
from tensorflow.python.framework.ops import disable_eager_execution

from src.operators.NUFFT2D import NUFFT2D
# disable_eager_execution()

import os
import sys 
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

# operators and sampling patterns
from src.operators.NUFFT2D_TF import NUFFT2D_TF
from src.operators.NNFFT2D_TF import NNFFT2D_TF
from src.operators.IdentityOperator import IdentityOperator

from src.sampling.uv_sampling import spider_sampling, random_sampling

 # some custom callbacks
from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 

# model and dataset generator
from src.networks.UNet import UNet
from src.dataset import Dataset, PregeneratedDataset, data_map, make_yogadl_dataset, measurement_func, random_crop, data_map_image

# selecting one gpu to train on
from src.util import gpu_setup
gpu_setup()


#TODO add a nice argument parser

epochs = 200
set_size = 2000 # size of the train set
save_freq = 20 # save every 20 epochs
batch_size = 20 
max_train_time = 40*60 # time after which training should stop in mins


ISNR = 30 #dB
network = "UNet"
activation = "linear"
load_weights = bool(1) # continuing the last run
operator = "NUFFT_SPIDER"

postfix = ""


data = "COCO"

Nd = (256, 256)
Kd = (512, 512)
Jd = (6,6)

input_type="measurements"
data_op = None
if operator == "NUFFT_SPIDER":
    uv = spider_sampling()
    y_shape = len(uv)
    op = NUFFT2D_TF

    # sampling density based weighting
    grid_cell = 2*np.pi /512 
    binned = (uv[:,:]+np.pi+.5*grid_cell) // grid_cell
    binned = [tuple(x) for x in binned]
    cells = set(binned)
    w_gridded = np.zeros(uv.shape[0])
    for cell in list(cells):
        mask = np.all(np.array(cell) ==  binned, axis=1)
        w_gridded[mask] = np.sum(mask)

    w = np.linalg.norm(uv, axis=1)
    # w = 1/w_gridded
    w /= w.max()
elif operator == "NUFFT_Random":
    y_shape = int(Nd[0]**2/2)
    uv = random_sampling(y_shape)
    op = NUFFT2D_TF
    w = np.ones(len(uv)) # no weights necessary for 50% sampling
elif operator == "NNFFT_Random":
    y_shape = int(Nd[0]**2/2)
    uv = random_sampling(y_shape)
    op = NNFFT2D_TF
    w = np.ones(len(uv)) # no weights necessary for 50% sampling
elif operator == "Identity":
    y_shape = Nd
    op = IdentityOperator
    data_op = IdentityOperator
    input_type="image"
    w = 1
    uv = None
    ISNR=24 # more noise for image domain
else:
    print("No such operator")
    exit()


# In[4]:


y_dirty_test = np.load(f"./data/intermediate/COCO/NUFFT_SPIDER/y_dirty_test_{ISNR}dB.npy").reshape(-1,y_shape)


# In[5]:


batch_size = 1


# # PseudoInverse

# In[24]:


tf.keras.backend.clear_session()


# In[25]:


from src.networks.PseudoInverse import PseudoInverse


# In[26]:


pi = PseudoInverse(
    Nd, 
    uv=uv,
    op=op, 
    measurement_weights=w,
    batch_size=batch_size,
    rescale=False
    )


# In[27]:


st = time.time()
pi.predict(y_dirty_test, batch_size=batch_size)
print(f"completed in {time.time()-st:.2f}s ({(time.time()-st)/len(y_dirty_test)*1e3:.1f} ms per sample)")


# In[28]:


del pi


# # UNet

# In[11]:


tf.keras.backend.clear_session()


# In[12]:


unet = UNet(
    Nd, 
    uv=uv,
    op=op, 
    depth=4, 
    conv_layers=2,
    input_type=input_type, 
    measurement_weights=w,
    batch_size=batch_size,
    residual=True
    )


# In[13]:


checkpoint_folder = f"./models/{data}/{operator}/{network}_{ISNR}dB{postfix}"

latest = tf.train.latest_checkpoint(checkpoint_folder)
unet.load_weights(latest)


# In[14]:


st = time.time()
unet.predict(y_dirty_test, batch_size=batch_size)
print(f"completed in {time.time()-st}s ({(time.time()-st)/len(y_dirty_test)*1e3} ms per sample)")


# In[15]:


del unet


# # GUNet

# In[16]:


tf.keras.backend.clear_session()


# In[17]:


from src.networks.GUnet import GUnet


# In[20]:


tf.compat.v1.disable_eager_execution() # GUNet cannot use eager execution


# In[21]:


gunet = GUnet(
    Nd, 
    uv=uv,
    op=op, 
    depth=4, 
    conv_layers=2,
    input_type=input_type, 
    measurement_weights=w,
    batch_size=batch_size,
    residual=True
    )


# In[22]:


network = "GUnet"

checkpoint_folder = f"./models/{data}/{operator}/{network}_{ISNR}dB{postfix}"
latest = tf.train.latest_checkpoint(checkpoint_folder)
gunet.load_weights(latest)


# In[23]:


st = time.time()
gunet.predict(y_dirty_test, batch_size=batch_size)
print(f"completed in {time.time()-st}s ({(time.time()-st)/len(y_dirty_test)*1e3} ms per sample)")


# In[ ]:


del gunet


# In[ ]:



