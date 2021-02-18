
import numpy as np
from pynufft import NUFFT

import optimusprimal.primal_dual as primal_dual
import optimusprimal.linear_operators as linear_operators
import optimusprimal.prox_operators as prox_operators


class MeasurementOperator():
    """Simple measurement operator using the pyNUFFT package to sample at non-uniform u,v coordinates"""
    def __init__(self, vis, Nd=(256,256), Kd=(512,512), Jd=(6,6)):
        """ Initialises the measurement operators for u,v coordinates 'vis' (M x 2)"""
        self.op = NUFFT()
        self.op.plan(vis, Nd, Kd, Jd)
    
    def dir_op(self, x):
        y = self.op.forward(x) 
        return y
    
    def adj_op(self, y):
        x = self.op.adjoint(y)
        return x
    
    def self_adj(self, x):
        y = self.dir_op(x)
        x = self.adj_op(y)
        return x.real

class MeasurementOperatorDiscrete():
    """Simple measurement operator. This operator maps the u,v coordinates to the nearest gridpoint. """
    def __init__(self, vis, Nd=(256,256), Kd=(512,512)):
        """ Initialises the measurement operators for u,v coordinates 'vis' (M x 2)"""

        self.Nd = Nd
        self.Kd = Kd
        self.pupil = np.zeros(Kd)
        for u,v in vis:
            self.pupil[ int(u/np.pi*Kd[0]//2 -.01)+Kd[0]//2, int(v/np.pi*Kd[1]//2-.01) + Kd[1]//2] = 1

    def dir_op(self, x):
        # zero pad in real space for upsampled k-space
        y = self._fft(np.pad(x, (
            ( (self.Kd[0]-self.Nd[0])//2, (self.Kd[0]-self.Nd[0])//2),
            ( (self.Kd[1]-self.Nd[1])//2, (self.Kd[1]-self.Nd[1])//2)
            ))) * (self.pupil) # measure only the given visabilities
        return y
    
    def adj_op(self, y):
        x = self._ifft(y*self.pupil) # use only given visabilities for reconstruction
        x = x[
            (self.Kd[0]-self.Nd[0])//2: (self.Kd[0]-self.Nd[0])//2 +self.Nd[0],
            (self.Kd[1]-self.Nd[1])//2: (self.Kd[1]-self.Nd[1])//2 +self.Nd[1]
            ] # remove zero padding from image
        return x
    
    def _fft(self, x):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x)))

    def _ifft(self, Fx):
        return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Fx))))

    def self_adj(self, x):
        y = self.dir_op(x)
        x = self.adj_op(y)
        return x


def RadialSampling(n_spokes=37, n_bins=10):
    """Returns the u,v coordinates for uniform sampling on radial spokes. Normalised to [-pi, pi]"""
    r   = np.linspace( np.pi, 0,  n_bins, endpoint=False)[::-1]
    phi = np.linspace(0, 2*np.pi, n_spokes,   endpoint=False)

    rr, pp = np.meshgrid(r, phi)
    
    vis = np.array([(rr*np.sin(pp)).reshape(-1), (rr*np.cos(pp)).reshape(-1)]).T
    return vis/np.max(vis) *np.pi


def SpiderSampling(n_spokes=37, n_lenslets=24, spectral_bins=10, max_baseline=0.65, min_baseline=0.15):
    """Returns the u,v coordinates for sampling following the SPIDER instrument. Normalised to [-pi, pi] (as described in Liu et al. 2018) """

    coordinates = np.array([ ( (lens+1)/n_lenslets * (max_baseline-min_baseline) + min_baseline)* np.array([np.cos(2*np.pi *spoke/n_spokes), np.sin(2*np.pi *spoke/n_spokes)]) for spoke in range(n_spokes) for lens in range(n_lenslets)])
    spectrals = np.linspace(500e-9, 900e-9, spectral_bins)

    vis = list()
    for n in range(n_spokes):
        for i in range(n_lenslets//2):
            x1 = coordinates[n*n_lenslets + i]
            x2 = coordinates[(n+1)*n_lenslets -i -1]
            for k in range(spectral_bins):
                vis.append(2*np.pi*(x1-x2)/spectrals[k])

    vis = np.array(vis)
    vis = vis / np.max( np.abs(vis)) *np.pi # normalise to [-pi, pi]
    return vis


def SpiderSampling2(n_spokes=37, n_lenslets=24, spectral_bins=10, max_baseline=0.65, min_baseline=0.15):
    """Returns the u,v coordinates for sampling following the SPIDER instrument. Normalised to [-pi, pi] (as described in Duncan et al. 2001) """

    baselines = np.array([3.6, 7.2, 10.8, 14.4, 18.0, 21.6, 28.8, 36.0, 46.8, 61.2, 79.2, 104.4]) # predefined baseline lengths in mm (Duncan et al. 2001)
   
    spectral_bins = np.linspace(500e-9, 900e-9, spectral_bins) /1e3 #mm

    vis = list()
    for n in range(n_spokes):
        angle = n/n_spokes * np.pi *2
        for i in range(len(baselines)):
            for k in range(len(spectral_bins)):
                l = baselines[i]/spectral_bins[k]
                vis.append((l*np.sin(angle), l*np.cos(angle))) 
    
    vis = np.array(vis)
    vis = vis / np.max( np.abs(vis)) *np.pi # normalise to [-pi, pi]
    return vis
    

def solver(y, phi=None):

    x_init = phi.adj_op(y)

    options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False}
    ISNR = 20.
    sigma = 10**(-ISNR / 20.)
    size = len(y)

    epsilon = np.sqrt(size + 2. * np.sqrt(2*size)) * sigma

    # proximal operator
    p = prox_operators.l2_ball( epsilon, y, phi)
    nu, sol = linear_operators.power_method(phi, np.ones_like(x_init))
    p.beta = nu

    # sparsity 
    wav = ['db' + str(i) for i in range(1,8)]
    levels = 4
    shape = (size,)
    psi = linear_operators.dictionary(wav, levels, x_init.shape)

    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(x_init))) * 1e-3, psi)
    h.beta = 1.

    # f = prox_operators.real_prox()
    f = None


    z, diag = primal_dual.FBPD(x_init, options, None, f, h, p)

    return z