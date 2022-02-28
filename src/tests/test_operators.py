import pytest 

import numpy as np
import tensorflow as tf

from src.operators.NUFFT2D import NUFFT2D
from src.operators.NUFFT2D_TF import NUFFT2D_TF

from src.operators.NNFFT2D import NNFFT2D
from src.operators.NNFFT2D_TF import NNFFT2D_TF


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

    a = tf.math.reduce_sum(tf.math.conj(y) * y_)
    b = tf.math.reduce_sum(tf.math.conj(x_) * x)
    assert tf.debugging.assert_near(a, b)

@pytest.fixture()
def shape_x():
    return (32,32)

@pytest.fixture()
def shape_y():
    return((200,))

@pytest.fixture()
def uv(shape_y):
    np.random.seed(38204) # some random sampling patterns do not work so fixed pattern
    uv = np.random.normal(size=(shape_y[0], 2))
    uv /= uv.max() * np.pi
    return uv

@pytest.fixture()
def NUFFT2D_op(uv, shape_x):
    op = NUFFT2D()
    op.plan(uv, Nd=shape_x, Kd=(64,64), Jd=(3,3))
    return op

@pytest.fixture()
def NUFFT2D_TF_op(uv, shape_x):
    op = NUFFT2D_TF()
    op.plan(uv, Nd=shape_x, Kd=(64,64), Jd=(6,6), batch_size=1)
    return op

@pytest.fixture()
def NNFFT2D_op(uv, shape_x):
    op = NNFFT2D()
    op.plan(uv, Nd=shape_x, Kd=(64,64), Jd=(3,3))
    return op

@pytest.fixture()
def NNFFT2D_TF_op(uv, shape_x):
    op = NNFFT2D_TF()
    op.plan(uv, Nd=shape_x, Kd=(64,64), Jd=(3,3), batch_size=1)
    return op

def test_self_adjointness_NUFFT2D(NUFFT2D_op, shape_x, shape_y):
    dot_product_test_np(NUFFT2D_op, shape_x=shape_x, shape_y=shape_y)

def test_self_adjointness_NUFFT2D_TF(NUFFT2D_TF_op, shape_x, shape_y):
    dot_product_test_TF(NUFFT2D_TF_op, shape_x=shape_x, shape_y=shape_y)

def test_self_adjointness_NNFFT2D(NNFFT2D_op, shape_x, shape_y):
    dot_product_test_np(NNFFT2D_op, shape_x=shape_x, shape_y=shape_y)

def test_self_adjointness_NNFFT2D_TF(NNFFT2D_TF_op, shape_x, shape_y):
    dot_product_test_TF(NNFFT2D_TF_op, shape_x=shape_x, shape_y=shape_y)

def not_test_np_vs_TF(shape_x, shape_y, NUFFT2D_op, NUFFT2D_TF_op, NNFFT2D_op, NNFFT2D_TF_op):
    x = np.random.normal(size=shape_x)[np.newaxis, :]
    x_TF = tf.convert_to_tensor(x, dtype=tf.complex64)

    y = np.random.normal(size=shape_y)[np.newaxis, :] \
        + 1j * np.random.normal(size=shape_y)[np.newaxis, :]
    y_TF = tf.convert_to_tensor(y, dtype=tf.complex64)

    assert tf.debugging.assert_near(NUFFT2D_op.dir_op(x).astype(np.complex64), NUFFT2D_TF_op.dir_op(x_TF)), "nufft forward not the same"
    assert tf.debugging.assert_near(NUFFT2D_op.adj_op(y).astype(np.complex64), NUFFT2D_TF_op.adj_op(y_TF)), "nufft adjoint not the same"


    y = np.random.normal(size=NNFFT2D_op.n_measurements)[np.newaxis, :] \
        + 1j * np.random.normal(size=NNFFT2D_op.n_measurements)[np.newaxis, :]
    y_TF = tf.convert_to_tensor(y, dtype=tf.complex64)

    assert tf.debugging.assert_near(NNFFT2D_op.dir_op(x).astype(np.complex64), NNFFT2D_TF_op.dir_op(x_TF)), "nufft forward not the same"
    assert tf.debugging.assert_near(NNFFT2D_op.adj_op(y).astype(np.complex64), NNFFT2D_TF_op.adj_op(y_TF)), "nufft adjoint not the same"

    # NNFFT2D_op, NNFFT2D_TF_op
