import numpy as np

from src.networks.UNet import UNet
from src.operators.NUFFT2D_TF import NUFFT2D_TF
from src.networks.HighLowPassNet import HighLowPassNet

from src.sampling.uv_sampling import random_sampling

def test_UNet():
    shape = (32,32)

    uv = np.random.normal(size=(200,2))
    uv *= np.pi / uv.max()

    measurement_weights = np.linalg.norm(uv, axis=1)
    measurement_weights /= measurement_weights.max()

    m = UNet(shape, uv, op=NUFFT2D_TF, measurement_weights=measurement_weights, input_type="measurements")
    # print(m.summary())

# test_Unet()

def test_highlow():
    shape = (32,32)
    # print(m.summary())


    uv = random_sampling(200)

    measurement_weights = np.linalg.norm(uv, axis=1)
    measurement_weights /= measurement_weights.max()

    m = HighLowPassNet(shape, uv=uv, op=NUFFT2D_TF, measurement_weights=measurement_weights)
    # print(m.summary())


def test_GUNet():
    shape = (32,32)
    
    uv = random_sampling(200)

    measurement_weights = np.linalg.norm(uv, axis=1)
    measurement_weights /= measurement_weights.max()

    m = HighLowPassNet(shape, uv=uv, op=NUFFT2D_TF, measurement_weights=measurement_weights)
    # print(m.summary())