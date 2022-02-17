import numpy as np
import tensorflow as tf

from src.operators.NUFFT2D import NUFFT2D
from src.operators.NUFFT2D_TF import NUFFT2D_TF

from src.operators.NNFFT2D import NNFFT2D
from src.operators.NNFFT2D_TF import NNFFT2D_TF

np.random.seed(42) # some random sampling patterns do not work so fixed pattern

def dot_product_test_np(op, shape_x, shape_y):
    x = np.random.normal(size=shape_x)
    y = np.random.normal(size=(op.n_measurements,)) \
        + 1j * np.random.normal(size=(op.n_measurements,))
    
    y_ = op.dir_op(x)
    x_ = op.adj_op(y)
    
    a = np.sum(np.conjugate(y) * y_)
    b = np.sum(np.conjugate(x_) * x)
    assert np.allclose(a, b)


def dot_product_test_TF(op, shape_x, shape_y):
    x = np.random.normal(size=shape_x)[np.newaxis,:]
    y = np.random.normal(size=(op.n_measurements,)) \
        + 1j * np.random.normal(size=(op.n_measurements,))[np.newaxis,:]
    
    x = tf.convert_to_tensor(x, dtype=tf.complex64)
    y = tf.convert_to_tensor(y, dtype=tf.complex64)
    
    y_ = op.dir_op(x)
    x_ = op.adj_op(y)

    a = np.sum(np.conjugate(y.numpy()) * y_.numpy())
    b = np.sum(np.conjugate(x_.numpy()) * x.numpy())
    assert np.allclose(a, b)

def test_self_adjointness_NUFFT2D():
    shape_x = (32,32)
    shape_y = (200,)
    uv = np.random.normal(size=(shape_y[0], 2))
    uv /= uv.max() * np.pi
    op = NUFFT2D()
    op.plan(uv, Nd=shape_x, Kd=(64,64), Jd=(3,3))

    dot_product_test_np(op, shape_x=shape_x, shape_y=shape_y)

def test_self_adjointness_NUFFT2D_TF():
    shape_x = (32,32)
    shape_y = (200,)
    uv = np.random.normal(size=(shape_y[0], 2))
    uv /= uv.max() * np.pi
    op = NUFFT2D_TF()
    op.plan(uv, Nd=shape_x, Kd=(64,64), Jd=(3,3), batch_size=1)

    dot_product_test_TF(op, shape_x=shape_x, shape_y=shape_y)

def test_self_adjointness_NNFFT2D():
    shape_x = (32,32)
    shape_y = (200,)
    uv = np.random.normal(size=(shape_y[0], 2))
    uv /= uv.max() * np.pi
    op = NNFFT2D()
    op.plan(uv, Nd=shape_x, Kd=(64,64), Jd=(3,3))

    dot_product_test_np(op, shape_x=shape_x, shape_y=shape_y)

def test_self_adjointness_NNFFT2D_TF():
    shape_x = (32,32)
    shape_y = (200,)
    uv = np.random.normal(size=(shape_y[0], 2))
    uv /= uv.max() * np.pi
    op = NNFFT2D_TF()
    op.plan(uv, Nd=shape_x, Kd=(64,64), Jd=(3,3), batch_size=1)

    dot_product_test_TF(op, shape_x=shape_x, shape_y=shape_y)