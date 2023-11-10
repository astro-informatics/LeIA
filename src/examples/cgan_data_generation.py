#!/usr/bin/env python

import os
import glob
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from src.operators.NUFFT2D import NUFFT2D


Nd = (360, 360)

# Kd = (720, 720)
Kd = (360, 360) # TODO think about upsampling

Jd = (6, 6)
batch_size = 1

uv = np.load("./data/intermediate/TNG/NUFFT_Random_var/uv_big.npy")
uv_original = np.load("./data/intermediate/TNG/NUFFT_Random_var/uv_original.npy")
sel_original = np.load("./data/intermediate/TNG/NUFFT_Random_var/sel.npy")

m_op = NUFFT2D()
m_op.plan(uv, Nd, Kd, Jd, batch_size)


m_op_original = NUFFT2D()
m_op_original.plan(uv[sel_original], Nd, Kd, Jd, batch_size)


m_ops = []
grids = []
sels = []
for i in range(100):
    m_op2 = NUFFT2D()
    sel = np.random.permutation(len(uv)) < len(uv) // 2
    m_op2.plan(uv[sel], Nd, Kd, Jd, batch_size)

    m_ops.append(deepcopy(m_op2))
    sels.append(sel)
    grids.append(m_op2._k2kk(np.ones(len(uv[sel]))).real)


data_folder = "/share/gpu0/mars/TNG_data/processed_360"
im_paths = glob.glob(data_folder + "/*.npy")


x_true = []
ys = []
y_grid_fixed = []
grid_fixed = []
y_grid_random = []
grid_random = []
sel_random = []

x_dirty = []
x_dirty_random = []

for idx, im_path in enumerate(tqdm.tqdm(im_paths)):
    im = np.load(im_path)
    x_true.append(im)
    y_ = m_op.dir_op(im)
    ys.append(y_)
    y_grid_fixed.append(m_op_original._k2kk(y_[sel_original])[0])
    if idx == 0:
        fixed_grid = m_op_original._k2kk(np.ones(np.sum(sel_original)))[0]
    grid_fixed.append(fixed_grid)

    sel_random = sels[idx % len(m_ops)]
    y_grid_random.append(m_ops[idx % len(m_ops)]._k2kk(y_[sel_random])[0])
    grid_random.append(grids[idx % len(m_ops)])

    if idx % 1000 == 0:
        save_dir = "/share/gpu0/mars/TNG_data/preprocessed_360/"
        os.makedirs(save_dir, exist_ok=True)
        np.save(save_dir + "./x_true.npy", np.array(x_true))
        np.save(save_dir + "./y_nu.npy", np.array(ys))
        np.save(save_dir + "./y_gridded.npy", np.array(y_grid_fixed))
        np.save(save_dir + "./uv_gridded.npy", np.array(grid_fixed).real)
        np.save(save_dir + "./y_gridded_random.npy", np.array(y_grid_random))
        np.save(save_dir + "./uv_gridded_random.npy", np.array(grid_random).real)
        np.save(save_dir + "./uv_sel_random.npy", np.array(sel_random))

#            if k.ndim == 1:
#             k = k[np.newaxis, :]
#         kk = self._k2kk(k)
save_dir = "/share/gpu0/mars/TNG_data/preprocessed_360/"
os.makedirs(save_dir, exist_ok=True)
np.save(save_dir + "./x_true.npy", np.array(x_true))
np.save(save_dir + "./y_nu.npy", np.array(ys))
np.save(save_dir + "./y_gridded.npy", np.array(y_grid_fixed))
np.save(save_dir + "./uv_gridded.npy", np.array(grid_fixed).real)
np.save(save_dir + "./y_gridded_random.npy", np.array(y_grid_random))
np.save(save_dir + "./uv_gridded_random.npy", np.array(grid_random).real)
np.save(save_dir + "./uv_sel_random.npy", np.array(sel_random))
