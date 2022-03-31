import numpy as np
import tensorflow as tf

import pytest 

from src.operators.NUFFT2D_TF import NUFFT2D_TF
from src.networks.UNet import UNet
from src.networks.GUnet import GUnet
from src.networks.GUnet_variant import GUnet_variant
from src.networks.HighLowPassNet import HighLowPassNet
from src.networks.LFB import LFB

from src.sampling.uv_sampling import random_sampling

@pytest.fixture()
def x():
    return np.random.random(size=(1, 32,32))

@pytest.fixture()
def y():
    return np.random.random(size=(1, 200))

def network_test(network, x, y):
    tf.keras.backend.clear_session()
    x_shape = (32,32)
    y_shape = (200,)

    uv = np.random.normal(size=(y_shape[0],2))
    uv *= np.pi / uv.max() * 0.9

    measurement_weights = np.linalg.norm(uv, axis=1)
    measurement_weights /= measurement_weights.max()

    model = network(
        x_shape, 
        uv, 
        op=NUFFT2D_TF, 
        measurement_weights=measurement_weights, 
        input_type="measurements", 
        batch_size=1
        )

    model.fit(y, x)

    new_uv = np.random.normal(size=(y_shape[0],2))
    new_uv *= np.pi / new_uv.max() * 0.9   
    
    model = model.rebuild_with_op(new_uv)
    model.fit(y, x)

    # TODO add tests that assert the right input and output shapes
    # TODO add tests that check shapes of measurement weights not being the same
    # TODO add test for the rebuild with op function

def test_UNet(x, y):
    network_test(UNet, x, y)

def test_GUNet(x, y):
    tf.compat.v1.disable_eager_execution()
    network_test(GUnet, x, y)

def test_HighLowPassNet(x, y):
    tf.compat.v1.disable_eager_execution()
    network_test(HighLowPassNet, x, y)

def test_GUNet_variant(x,y):
    tf.compat.v1.disable_eager_execution()
    network_test(GUnet_variant, x, y)

# def test_LFB():
#     #TODO implement this test to work, currently doesn't work since the other functions require eager execution disabled and this requires it enabled
#     pass
#     # tf.compat.v1.enable_eager_execution()
#     # shape = (32,32)
    
#     # uv = random_sampling(200)

#     # measurement_weights = np.linalg.norm(uv, axis=1)
#     # measurement_weights /= measurement_weights.max()

#     # m = LFB(shape, uv=uv, op=NUFFT2D_TF, measurement_weights=measurement_weights)
#     # # print(m.summary())