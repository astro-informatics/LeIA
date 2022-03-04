import numpy as np
import tensorflow as tf

from src.operators.NUFFT2D_TF import NUFFT2D_TF
from src.networks.UNet import UNet
from src.networks.GUnet import GUnet
from src.networks.GUnet_variant import GUnet_variant2
from src.networks.HighLowPassNet import HighLowPassNet
from src.networks.LFB import LFB

from src.sampling.uv_sampling import random_sampling

def test_UNet():
    tf.compat.v1.disable_eager_execution()
    shape = (32,32)

    uv = np.random.normal(size=(200,2))
    uv *= np.pi / uv.max()

    measurement_weights = np.linalg.norm(uv, axis=1)
    measurement_weights /= measurement_weights.max()

    m = UNet(shape, uv, op=NUFFT2D_TF, measurement_weights=measurement_weights, input_type="measurements")
    # print(m.summary())

# test_Unet()

def test_highlow():
    tf.compat.v1.disable_eager_execution()
    shape = (32,32)
    # print(m.summary())


    uv = random_sampling(200)

    measurement_weights = np.linalg.norm(uv, axis=1)
    measurement_weights /= measurement_weights.max()

    m = HighLowPassNet(shape, uv=uv, op=NUFFT2D_TF, measurement_weights=measurement_weights)
    # print(m.summary())


def test_GUNet():
    tf.compat.v1.disable_eager_execution()
    shape = (32,32)
    
    uv = random_sampling(200)

    measurement_weights = np.linalg.norm(uv, axis=1)
    measurement_weights /= measurement_weights.max()

    m = GUnet(shape, uv=uv, op=NUFFT2D_TF, measurement_weights=measurement_weights)
    # print(m.summary())

def test_GUNet_variant():
    tf.compat.v1.disable_eager_execution()
    shape = (32,32)
    
    uv = random_sampling(200)

    measurement_weights = np.linalg.norm(uv, axis=1)
    measurement_weights /= measurement_weights.max()

    m = GUnet_variant.GUnet_variant2(shape, uv=uv, op=NUFFT2D_TF, measurement_weights=measurement_weights)
    # print(m.summary())

def test_LFB():
    tf.compat.v1.disable_eager_execution()
    shape = (32,32)
    
    uv = random_sampling(200)

    measurement_weights = np.linalg.norm(uv, axis=1)
    measurement_weights /= measurement_weights.max()

    m = LFB(shape, uv=uv, op=NUFFT2D_TF, measurement_weights=measurement_weights)
    # print(m.summary())