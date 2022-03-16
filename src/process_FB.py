# __requires__= 'scipy==1.6.3'
# import pkg_resources
# pkg_resources.require('scipy==1.6.3')
from ast import operator
import scipy

import numpy as np
import pickle 
import os
import sys
import multiprocessing
import time 
import tqdm
import tensorflow as tf
import copy

from src.sampling.uv_sampling import spider_sampling, random_sampling
from src.operators.NUFFT2D_TF import NUFFT2D_TF
from src.operators.NNFFT2D import NNFFT2D
from src.solvers import PrimalDual_l1_constrained, FB_unconstrained
from src.operators.dictionary import wavelet_basis
from src.networks.UNet import UNet

import optimusprimal.forward_backward as forward_backward
import optimusprimal.grad_operators as grad_operators
import optimusprimal.ai_operators as ai_operators
import optimusprimal.prox_operators as prox_operators



ISNR = 15 #int(sys.argv[1])
mode = sys.argv[1] # train or test
data = "COCO" #sys.argv[3]
pnp = bool(int(sys.argv[2]))
project_folder = os.environ["HOME"] +"/src_aiai/"
# ISNR = 50
# mode = 'train'
# mode = 'test'

Nd = (256, 256)
Kd = (512, 512)
Jd = (6,6)

batch_size = 1
operator = "NUFFT_Random"
y_shape = int(Nd[0]**2/2)
uv = random_sampling(y_shape)
m_op = NUFFT2D_TF()
m_op.plan(uv, Nd, Kd, Jd, batch_size=batch_size)
iterations = 200

class l1_norm(prox_operators.ProximalOperator):
    """This class computes the proximity operator of the l2 ball.

                        f(x) = ||Psi x||_1 * gamma

    When the input 'x' is an array. gamma is a regularization term. Psi is a sparsity operator.
    """

    def __init__(self, gamma, Psi=None):
        """Initialises an l1-norm proximal operator class

        Args:

            gamma (double >= 0): Regularisation parameter
            Psi (Linear operator): Regularisation functional (typically wavelets)

        Raises:

            ValueError: Raised if regularisation parameter is not postitive semi-definite
        """

        if np.any(gamma <= 0):
            raise ValueError("'gamma' must be positive semi-definite")

        self.gamma = gamma
        self.beta = 1.0

        if Psi is None:
            self.Psi = linear_operators.identity()
        else:
            self.Psi = Psi

    def prox(self, x, tau):
        """Evaluates the l1-norm prox of x

        Args:

            x (np.ndarray): Array to evaluate proximal projection of
            tau (double): Custom weighting of l1-norm prox

        Returns:

            l1-norm prox of x
        """
        return np.maximum(0, np.abs(x) - self.gamma * tau) * np.exp(
            complex(0, 1) * np.angle(x)
        )

    def fun(self, x):
        """Evaluates loss of functional term of l1-norm regularisation

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            l1-norm loss
        """
        return np.abs(self.gamma * x).sum()

    def dir_op(self, x):
        """Evaluates the forward regularisation operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward regularisation operator applied to x
        """
        return self.Psi.dir_op(x[0])

    def adj_op(self, x):
        """Evaluates the forward adjoint regularisation operator

        Args:

            x (np.ndarray): Array to adjoint transform

        Returns:

            Forward adjoint regularisation operator applied to x
        """
        return self.Psi.adj_op(x)[None, :]

sigma = 0.05

if pnp:
    denoiser = UNet(
        Nd, 
        uv=None,
        op=None, 
        depth=4, 
        input_type="image", 
        measurement_weights=1,
        batch_size=1
    )
    latest = tf.train.latest_checkpoint(f"./models/COCO/Identity/UNet_15dB")
    denoiser.load_weights(latest)
    h = ai_operators.PnpDenoiser(denoiser, sigma)
    postfix = "_PnP"
else:
    step = 5e-3
    psi = wavelet_basis(Nd)
    h = l1_norm(step, psi)
    postfix = ""

if mode == "train":
    x_true = np.load(project_folder + f"data/intermediate/{data}/{operator}/x_true_train_{ISNR}dB.npy")
    y_dirty = np.load(project_folder +f"data/intermediate/{data}/{operator}/y_dirty_train_{ISNR}dB.npy")
    # noise_val = np.load(project_folder +f"data/intermediate/{data}/noise_levels_train_{ISNR}dB{postfix}.npy")
    try:
        predict_x = np.load(project_folder + f"data/processed/{data}/{operator}/PD_train_predict_{ISNR}dB{postfix}.npy")
        timings = np.load(project_folder + f"data/processed/{data}/{operator}/times_train_{ISNR}dB{postfix}.npy")
        diags = pickle.load(open(project_folder + f"results/{data}/{operator}/diag_{ISNR}dB{postfix}.npy", "rb"))
    except:
        predict_x = np.zeros_like(x_true)
        timings = np.zeros(len(predict_x))
        diags = [None] * len(predict_x)

    # ASSUMING WE KNOW THE NOISE VALUE

elif mode == "test":
    x_true = np.load(project_folder + f"data/intermediate/{data}/{operator}/x_true_test_{ISNR}dB.npy")
    y_dirty = np.load(project_folder + f"data/intermediate/{data}/{operator}/y_dirty_test_{ISNR}dB.npy")
    # noise_val = np.load(project_folder +f"data/intermediate/{data}/noise_levels_test_{ISNR}dB{postfix}.npy")
    #predict_x = np.zeros_like(x_true)
    try:
        predict_x = np.load(project_folder + f"data/processed/{data}/{operator}/PD_test_predict_{ISNR}dB{postfix}.npy")
        timings = np.load(project_folder + f"data/processed/{data}/{operator}/times_test_{ISNR}dB{postfix}.npy")
    except:
        predict_x = np.zeros_like(x_true)
        timings = np.zeros(len(predict_x))
        diags = [None] * len(predict_x)

        
# uv = spider_sampling()
# m_op = NUFFT_op()
# m_op.plan(uv, (256,256), (512, 512), (6,6))




options = {"tol": 1e-5, "iter": iterations, "update_iter": 10, "record_iters": False, "real": False, "positivity": False}
solver = FB_unconstrained(m_op, h, options=options)

#

def process(iterable):
    y, m_op, noise, solver = iterable
    st = time.time()
    if noise <= 0:
        return np.zeros((256,256)), time.time() - st
    z, diag = solver.solve(y, m_op, noise)
    return  z, time.time() - st, diag

start = 0

start = np.where( np.sum(predict_x, axis=(1,2)) == 0 )[0][0]

os.makedirs(project_folder + f"data/processed/{data}/{operator}/", exist_ok=True)
os.makedirs(project_folder + f"results/{data}/{operator}", exist_ok=True)

noise_val = 0.05 # calculated from train set 
pool_size = 100

# solvers = [copy.deepcopy(solver) for i in range(pool_size)]

for i in tqdm.tqdm(range(start, len(x_true), pool_size)):
    iterables = [(y_dirty[i].reshape(1, -1), m_op, noise_val, solver) for i in range(i, min(i+pool_size, len(y_dirty)))]
    # with multiprocessing.Pool(pool_size) as pool:
    #     result = pool.map(process, iterables)
    result = [process(x) for x in tqdm.tqdm(iterables)]
    a = list(zip(*result)) 
    prediction = np.array(a[0]).reshape(-1,256,256)
    times = np.array(a[1]).reshape(-1)
    predict_x[i:i+len(prediction)] = prediction
    timings[i:i+len(times)] = times
    diags[i:i+len(times)] = a[2]

    if mode == "train":
        np.save(project_folder + f"data/processed/{data}/{operator}/train_predict_FB_{ISNR}dB{postfix}.npy", predict_x)
        np.save(project_folder + f"data/processed/{data}/{operator}/times_train_{ISNR}dB{postfix}.npy", timings)
        pickle.dump(diags, open(project_folder + f"results/{data}/{operator}/diag_{ISNR}dB{postfix}.npy", "wb"))
    elif mode == "test":
        np.save(project_folder + f"data/processed/{data}/{operator}/test_predict_FB_{ISNR}dB{postfix}.npy", predict_x)
        np.save(project_folder + f"data/processed/{data}/{operator}/times_test_{ISNR}dB{postfix}.npy", timings)

if mode == "train":
    np.save(project_folder + f"data/processed/{data}/{operator}/train_predict_FB_{ISNR}dB{postfix}.npy", predict_x)
    np.save(project_folder + f"data/processed/{data}/{operator}/times_train_{ISNR}dB{postfix}.npy", timings)
elif mode == "test":
    np.save(project_folder + f"data/processed/{data}/{operator}/test_predict_FB_{ISNR}dB{postfix}.npy", predict_x)
    np.save(project_folder + f"data/processed/{data}/{operator}/times_test_{ISNR}dB{postfix}.npy", timings)
