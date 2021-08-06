__requires__= 'scipy==1.6.3'
import pkg_resources
pkg_resources.require('scipy==1.6.3')
import scipy

import numpy as np
import pickle 
import os
import sys
import multiprocessing
import time 

from src.sampling.uv_sampling import spider_sampling
from src.operators.measurement import NUFFT_op
from src.solvers import PrimalDual_l1_constrained
from src.operators.dictionary import wavelet_basis

ISNR = int(sys.argv[1])
mode = sys.argv[2]
data = sys.argv[3]

project_folder = os.environ["HOME"] +"/src_aiai/"
# ISNR = 50
# mode = 'train'
# mode = 'test'

if mode == "train":
    x_true = np.load(project_folder + f"data/intermediate/{data}/x_true_train_{ISNR}dB.npy")
    y_dirty = np.load(project_folder +f"data/intermediate/{data}/y_dirty_train_{ISNR}dB.npy")
    noise_val = np.load(project_folder +f"data/intermediate/{data}/noise_levels_train_{ISNR}dB.npy")
    try:
        predict_x = np.load(project_folder + f"data/processed/{data}/PD_train_predict_{ISNR}dB.npy")
        timings = np.load(project_folder + f"data/processed/{data}/times_train_{ISNR}dB.npy")
    except:
        predict_x = np.zeros_like(x_true)
        timings = np.zeros(len(predict_x))

    # ASSUMING WE NOW THE NOISE VALUE

elif mode == "test":
    x_true = np.load(project_folder + f"data/intermediate/{data}/x_true_test_{ISNR}dB.npy")
    y_dirty = np.load(project_folder + f"data/intermediate/{data}/y_dirty_test_{ISNR}dB.npy")
    noise_val = np.load(project_folder +f"data/intermediate/{data}/noise_levels_test_{ISNR}dB.npy")
    #predict_x = np.zeros_like(x_true)
    try:
        predict_x = np.load(project_folder + f"data/processed/{data}/PD_test_predict_{ISNR}dB.npy")
        timings = np.load(project_folder + f"data/processed/{data}/times_test_{ISNR}dB.npy")
    except:
        predict_x = np.zeros_like(x_true)
        timings = np.zeros(len(predict_x))
        
        
uv = spider_sampling()
m_op = NUFFT_op(uv)

psi = wavelet_basis(x_true[0,:,:,0].shape)
solver = PrimalDual_l1_constrained(m_op=m_op, psi=psi, beta=1e-2,
    options={
        'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 
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
    return  z.real, time.time() - st

start = 0

start = np.where( np.sum(predict_x, axis=(1,2)) == 0 )[0][0]


for i in range(start, len(x_true), 30):
    iterables = [(y_dirty[i].reshape(-1), m_op, noise_val[i], solver) for i in range(i, min(i+30, len(y_dirty)))]
    with multiprocessing.Pool(30) as pool:
        result = pool.map(process, iterables)
    a = list(zip(*result)) 
    prediction = np.array(a[0]).reshape(-1,256,256,1)
    times = np.array(a[1]).reshape(-1)
    predict_x[i:i+len(prediction)] = prediction
    timings[i:i+len(times)] = times

    if mode == "train":
        np.save(project_folder + f"data/processed/{data}/PD_train_predict_{ISNR}dB.npy", predict_x)
        np.save(project_folder + f"data/processed/{data}/times_train_{ISNR}dB.npy", timings)
        
    elif mode == "test":
        np.save(project_folder + f"data/processed/{data}/PD_test_predict_{ISNR}dB.npy", predict_x)
        np.save(project_folder + f"data/processed/{data}/times_test_{ISNR}dB.npy", timings)

if mode == "train":
    np.save(project_folder + f"data/processed/{data}/PD_train_predict_{ISNR}dB.npy", predict_x)
    np.save(project_folder + f"data/processed/{data}/times_train_{ISNR}dB.npy", timings)
elif mode == "test":
    np.save(project_folder + f"data/processed/{data}/PD_test_predict_{ISNR}dB.npy", predict_x)
    np.save(project_folder + f"data/processed/{data}/times_test_{ISNR}dB.npy", timings)
