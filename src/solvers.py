#using code from Luke Pratley  - https://github.com/Luke-Pratley/Clear-Skies

import numpy as np

import logging
import numpy as np
import time


import optimusprimal.Empty as Empty
import optimusprimal.prox_operators as prox_operators
import optimusprimal.grad_operators as grad_operators
import optimusprimal.linear_operators as linear_operators
#import optimusprimal.primal_dual as primal_dual

logger = logging.getLogger('Optimus Primal')


def FBPD(x_init, options=None, g=None, f=None, h=None, p=None, r=None, viewer = None):
    if f is None:
        f = Empty.EmptyProx()
    if g is None:
        g = Empty.EmptyGrad()
    if h is None:
        h = Empty.EmptyProx()
    if p is None:
        p = Empty.EmptyProx()
    if r is None:
        r = Empty.EmptyProx()
    x = x_init
    y = h.dir_op(x) * 0.
    z = p.dir_op(x) * 0
    w = r.dir_op(x) * 0
    return FBPD_warm_start(x_init, y, z, w, options, g, f, h, p, r, viewer)

def FBPD_warm_start(x_init, y, z, w, options=None, g=None, f=None, h=None, p=None, r=None, viewer = None):
    """Takes in an input signal with proximal operators and a gradient operator
    and returns a solution with diagnostics."""
    # default inputs
    if f is None:
        f = Empty.EmptyProx()
    if g is None:
        g = Empty.EmptyGrad()
    if h is None:
        h = Empty.EmptyProx()
    if p is None:
        p = Empty.EmptyProx()
    if r is None:
        r = Empty.EmptyProx()
    if options is None:
        options = {'tol': 1e-4, 'iter': 500,
                   'update_iter': 100, 'record_iters': False}

    # checking minimum requrements for inputs
    assert hasattr(f, 'prox')
    assert hasattr(h, 'prox')
    assert hasattr(h, 'dir_op')
    assert hasattr(h, 'adj_op')
    assert hasattr(p, 'prox')
    assert hasattr(p, 'dir_op')
    assert hasattr(p, 'adj_op')
    assert hasattr(g, 'grad')

    # algorithmic parameters
    tol = options['tol']
    max_iter = options['iter']
    update_iter = options['update_iter']
    record_iters = options['record_iters']
    # step-sizes
    tau = 0.5 / g.beta
    sigmah = 1 * g.beta
    sigmap = 1 * g.beta
    sigmar = 1 * g.beta
    # initialization
    x = np.copy(x_init)

    logger.info('Running Forward Backward Primal Dual')
    timing = np.zeros(max_iter)
    criter = np.zeros(max_iter)
    hs, gs, fs, ps, rs, xs = [], [], [], [], [], []
    # algorithm loop
    for it in range(0, max_iter):

        t = time.time()
        # primal forward-backward step
        x_old = np.copy(x)
        x = x - tau * (g.grad(x) + h.adj_op(y) /
                       h.beta + p.adj_op(z) / p.beta + r.adj_op(w)/r.beta)
        x = f.prox(x, tau)
        # dual forward-backward step
        y = y + sigmah * h.dir_op(2 * x - x_old)
        y = y - sigmah * h.prox(y / sigmah, 1. / sigmah)

        z = z + sigmap * p.dir_op(2 * x - x_old)
        z = z - sigmap * p.prox(z / sigmap, 1. / sigmap)

        w = w + sigmar * r.dir_op(2 * x - x_old)
        w = w - sigmar * r.prox(w / sigmar, 1. / sigmar)
        # time and criterion
        if(record_iters):
            timing[it] = time.time() - t
            criter[it] = f.fun(x) + g.fun(x) + \
                h.fun(h.dir_op(x)) + p.fun(p.dir_op(x)) + r.fun(r.dir_op(x))
        if np.allclose(x, 0):
            x = x_old
            logger.info('[Primal Dual] converged to 0 in %d iterations', it)
            break
        # stopping rule
        if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old) and it > 10:
            logger.info('[Primal Dual] converged in %d iterations', it)
            break
        if(update_iter >= 0):
            if(it % update_iter == 0):
                logger.info('[Primal Dual] %d out of %d iterations, tol = %f',
                            it, max_iter, np.linalg.norm(x - x_old) / np.linalg.norm(x_old))
                if viewer is not None:
                    viewer(x, it)

                # ** Added some diagnostics to return
                hs.append(h.fun(h.dir_op(x)))
                gs.append(g.fun(x))
                fs.append(f.fun(x))
                ps.append(p.fun(p.dir_op(x)))
                rs.append(r.fun(r.dir_op(x)))
                xs.append(x)

        logger.debug('[Primal Dual] %d out of %d iterations, tol = %f',
                     it, max_iter, np.linalg.norm(x - x_old) / np.linalg.norm(x_old))

    criter = criter[0:it + 1]
    timing = np.cumsum(timing[0:it + 1])
    solution = x
    diagnostics = {'max_iter': it, 'times': timing, 'Obj_vals': criter, 'z': z, 'y': y, 'w': w, "hs": hs, "gs":gs, 'fs':fs, "ps":ps, "rs":rs, "xs":xs}
    return solution, diagnostics 



def l1_constrained_solver(data, phi, sigma, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve constrained l1 regularization problem
    """
    x_0 = phi.adj_op(data)
    size = len(np.ravel(data))
    epsilon = np.sqrt(size + 2 * np.sqrt(2 * size)) * sigma # lambda =2
    
    p = prox_operators.l2_ball(epsilon, data, phi)
    nu, sol = linear_operators.power_method(phi, np.ones_like(x_0))
    p.beta = nu    

    f = None

    if options['real'] == True:
        f = prox_operators.real_prox()
    if options["positivity"] == True:
        f = prox_operators.positive_prox()

    wav = ['db' + str(i) for i in range(1,8)]
    levels = 4
    psi = linear_operators.dictionary(wav, levels, x_0.shape)

    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(x_0))) * beta, psi)

    return FBPD(x_0, options, None, f, h, p)


def l1_unconstrained_solver(data, phi, sigma, beta=1e-3, options={'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False, 'positivity': False, 'real': False}):
    """
    Solve unconstrained l1 regularization problem
    """
    x_0 = phi.adj_op(data)

    g = grad_operators.l2_norm(sigma, data, phi)
    nu, sol = linear_operators.power_method(phi, np.ones_like(x_0))
    g.beta = nu / sigma**2

    if beta <= 0:
        h = None
    else:
        wav = ['db' + str(i) for i in range(1,8)]
        levels = 4
        psi = linear_operators.dictionary(wav, levels, x_0.shape)

        h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(x_0))) * beta, psi)

    f = None
    if options['real'] == True:
        if options["positivity"] == True:
            f = prox_operators.positive_prox()
        else:
            f = prox_operators.real_prox()

    return FBPD(x_0, options, g, f, h)


def wavelet_basis(shape, wavelets=range(1,8), levels=4):
    wav = ['db' + str(i) for i in wavelets]
    return linear_operators.dictionary(wav, levels, shape)