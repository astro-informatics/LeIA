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

import copy

from src.sampling.uv_sampling import spider_sampling, random_sampling
from src.operators.NUFFT2D import NUFFT2D
from src.operators.NNFFT2D import NNFFT2D
from src.solvers import PrimalDual_l1_constrained
from src.operators.dictionary import wavelet_basis

ISNR = 30 #int(sys.argv[1])
mode = sys.argv[1] # train or test
data = "COCO" #sys.argv[3]
operator = sys.argv[2]
project_folder = os.environ["HOME"] +"/src_aiai/"
# ISNR = 50
# mode = 'train'
# mode = 'test'

Nd = (256, 256)
Kd = (512, 512)
Jd = (6,6)

batch_size = 1
if operator == "NUFFT_SPIDER":
    uv = spider_sampling()
    y_shape = len(uv)
    m_op = NUFFT2D()
    m_op.plan(uv, Nd, Kd, Jd, batch_size=batch_size)
    iterations = 300
elif operator == "NUFFT_Random":
    y_shape = int(Nd[0]**2/2)
    uv = random_sampling(y_shape)
    m_op = NUFFT2D()
    m_op.plan(uv, Nd, Kd, Jd, batch_size=batch_size)
    iterations = 100
elif operator == "NNFFT_Random":
    y_shape = int(Nd[0]**2/2)
    uv = random_sampling(y_shape)
    m_op = NNFFT2D()
    m_op.plan(uv, Nd, Kd, Jd, batch_size=batch_size)
    iterations = 100


if mode == "train":
    x_true = np.load(project_folder + f"data/intermediate/{data}/{operator}/x_true_train_{ISNR}dB.npy")
    y_dirty = np.load(project_folder +f"data/intermediate/{data}/{operator}/y_dirty_train_{ISNR}dB.npy")
    noise_values = np.load(project_folder +f"data/intermediate/{data}/{operator}/noise_values_train_{ISNR}dB.npy")

    # noise_val = np.load(project_folder +f"data/intermediate/{data}/noise_levels_train_{ISNR}dB.npy")
    try:
        predict_x = np.load(project_folder + f"data/processed/{data}/{operator}/train_predict_PD_{ISNR}dB.npy")
        timings = np.load(project_folder + f"data/processed/{data}/{operator}/times_train_{ISNR}dB.npy")
        diags = pickle.load(open(project_folder + f"results/{data}/{operator}/diag_{ISNR}dB.npy", "rb"))
    except:
        predict_x = np.zeros_like(x_true)
        timings = np.zeros(len(predict_x))
        diags = [None] * len(predict_x)

    # ASSUMING WE KNOW THE NOISE VALUE

elif mode == "test":
    x_true = np.load(project_folder + f"data/intermediate/{data}/{operator}/x_true_test_{ISNR}dB.npy")
    y_dirty = np.load(project_folder + f"data/intermediate/{data}/{operator}/y_dirty_test_{ISNR}dB.npy")
    noise_values = np.load(project_folder +f"data/intermediate/{data}/{operator}/noise_values_test_{ISNR}dB.npy")

    # noise_val = np.load(project_folder +f"data/intermediate/{data}/noise_levels_test_{ISNR}dB.npy")
    #predict_x = np.zeros_like(x_true)
    try:
        predict_x = np.load(project_folder + f"data/processed/{data}/{operator}/test_predict_PD_{ISNR}dB.npy")
        timings = np.load(project_folder + f"data/processed/{data}/{operator}/times_test_{ISNR}dB.npy")
    except:
        predict_x = np.zeros_like(x_true)
        timings = np.zeros(len(predict_x))
        diags = [None] * len(predict_x)

        
# uv = spider_sampling()
# m_op = NUFFT_op()
# m_op.plan(uv, (256,256), (512, 512), (6,6))

psi = wavelet_basis(x_true[0,:].shape)
solver = PrimalDual_l1_constrained(m_op=m_op, psi=psi, beta=1e-5,
    options={
        'tol': 1e-6, 'iter': iterations, 'update_iter': 500, 
        'record_iters': False, 'positivity': True, 'real': True})


#for i in range(len(x_true)):
#    y = y_dirty[i].reshape(-1)
#    z, diag = solver.solve(y, m_op, noise_val[i])
#    predict_x[i,:,:,0] = z.real
#    if i%20 == 0:
#        if mode == "train":
#            np.save(project_folder + f"data/processed/{data}/PD_train_predict_{ISNR}dB.npy", predict_x)
#        elif mode == "test":
#            np.save(project_folder + f"data/processed/{data}/PD_test_predict_{ISNR}dB.npy", predict_x)
#

def process(iterable):
    y, m_op, noise, solver = iterable
    st = time.time()
    if noise <= 0:
        return np.zeros((256,256)), time.time() - st
    z, diag = solver.solve(y, m_op, noise)
    return  z.real, time.time() - st, diag

start = 0

start = np.where( np.sum(predict_x, axis=(1,2)) == 0 )[0][0]

os.makedirs(project_folder + f"data/processed/{data}/{operator}/", exist_ok=True)
os.makedirs(project_folder + f"results/{data}/{operator}", exist_ok=True)

if data == "COCO":
    noise_val = 0.05 # calculated from train set 
elif data == "SATS":
    noise_val = 0.04 # calculated from train set 
elif data == "TNG":
    noise_val = 0.02 # calculated from train set 
pool_size = 100

# solvers = [copy.deepcopy(solver) for i in range(pool_size)]

for i in tqdm.tqdm(range(start, len(x_true), pool_size)):
    iterables = [(y_dirty[i].reshape(-1), m_op, noise_values[i], solver) for i in range(i, min(i+pool_size, len(y_dirty)))]
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
        np.save(project_folder + f"data/processed/{data}/{operator}/train_predict_PD_{ISNR}dB.npy", predict_x)
        np.save(project_folder + f"data/processed/{data}/{operator}/times_train_{ISNR}dB.npy", timings)
        pickle.dump(diags, open(project_folder + f"results/{data}/{operator}/diag_{ISNR}dB.npy", "wb"))
    elif mode == "test":
        np.save(project_folder + f"data/processed/{data}/{operator}/test_predict_PD_{ISNR}dB.npy", predict_x)
        np.save(project_folder + f"data/processed/{data}/{operator}/times_test_{ISNR}dB.npy", timings)

if mode == "train":
    np.save(project_folder + f"data/processed/{data}/{operator}/train_predict_PD_{ISNR}dB.npy", predict_x)
    np.save(project_folder + f"data/processed/{data}/{operator}/times_train_{ISNR}dB.npy", timings)
elif mode == "test":
    np.save(project_folder + f"data/processed/{data}/{operator}/test_predict_PD_{ISNR}dB.npy", predict_x)
    np.save(project_folder + f"data/processed/{data}/{operator}/times_test_{ISNR}dB.npy", timings)
