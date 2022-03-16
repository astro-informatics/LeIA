import numpy as np

from optimusprimal import linear_operators, prox_operators, primal_dual, grad_operators, forward_backward


class BaseSolver():
    def __init__(self, m_op, options={}):
        self.m_op = m_op
        self.options=options

    def solve(self, y):
        return self.m_op.adj_op(y)



class PrimalDual_l1_constrained(BaseSolver):
    def __init__(self, m_op, psi, beta=1e-2, 
        options={ 'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 
        'record_iters': False, 'positivity': True, 'real': True}):
        super().__init__(m_op, options=options)
        self.psi = psi
        # self.noise_val = noise_val
        self.beta = beta
        print(self.options)

    def solve(self, y, m_op, noise_val):
        x0 = m_op.adj_op(y)

        size = len(np.ravel(y))
        epsilon = np.sqrt(size + 2 * np.sqrt(size)) * noise_val

        nu, sol = linear_operators.power_method(m_op, x0)

        p = prox_operators.l2_ball(epsilon, y, m_op)
        p.beta = nu
        
        step = np.max(np.real(self.psi.dir_op(x0))) * self.beta
        h = prox_operators.l1_norm(step, self.psi)
        f = None
        if self.options['real'] == True:
            if self.options["positivity"] == True:
                f = prox_operators.positive_prox()
            else:
                f = prox_operators.real_prox()
        r = None
        return primal_dual.FBPD(x0, self.options, None, f, h, p, r)


class FB_unconstrained(BaseSolver):
    def __init__(self, m_op, psi, beta=1e-2, 
        options={ 'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 
        'record_iters': False, 'positivity': True, 'real': True}):
        super().__init__(m_op, options=options)
        self.psi = psi
        # self.noise_val = noise_val
        self.beta = beta
        self.m_op = m_op
        print(self.options)


    def solve(self, y, m_op, sigma):
        x0 = self.m_op.adj_op(y)

        g = grad_operators.l2_norm_tf(sigma, y, self.m_op, self.m_op.Nd, (self.m_op.n_measurements,))

        f = None
        if self.options['real'] == True:
            if self.options["positivity"] == True:
                f = prox_operators.positive_prox()
            else:
                f = prox_operators.real_prox()


        best_estimate, diagnostics = forward_backward.FB(x_init=x0, alpha=0.5, options=self.options, f=f, g=g, h=self.psi)

        return best_estimate, diagnostics
