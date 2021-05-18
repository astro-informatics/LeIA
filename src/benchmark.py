import numpy as np
import time
import os
import pickle



def RSE(y, p):
    return np.sum(np.square(y-p))/np.sum(np.square(np.std(y)))


def generate_noisy_data(x_true, m_op, ISNR):    
    np.random.seed(42)

    y0 = m_op.dir_op(x_true)
    sigma = np.sqrt(np.mean(np.abs(y0)**2)) * 10**(-ISNR/20)
    n = np.random.normal(0, sigma, y0.shape) + 1j * np.random.normal(0, sigma, y0.shape)
    y = y0 + n
    x_dirty = m_op.adj_op(y)
    noise_val = np.std((m_op.adj_op(n)))
    return x_dirty, y, noise_val


def benchmark(x_true, solver, m_op, ISNR, save_name=None):
    if save_name:
        if os.path.exists(save_name + ".pkl"):
            results = pickle.load(open(save_name + ".pkl", "rb"))
            return results

    x_dirty, y, noise_val = generate_noisy_data(x_true, m_op, ISNR)

    start_time = time.time()
    sol, diag = solver.solve(y, m_op, noise_val)
    running_time = time.time() - start_time
    
    results = {
        "name": save_name,
        "x_true": x_true.real,
        "solution": sol.real,
        "ISNR": ISNR,
        "runtime": running_time,
        "RSE": RSE(x_true, sol.real),
    }
    if save_name:
        pickle.dump(results, open(save_name + ".pkl", "wb"))
    return results