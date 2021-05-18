
import numpy as np
from pynufft import NUFFT

from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule
import tensorflow as tf

class NUFFT_op():
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


class NUFFT_op_TF():
    """Simple measurement operator using the pyNUFFT package to sample at non-uniform u,v coordinates"""
    def __init__(self, vis, Nd=(256,256), Kd=(512,512), Jd=(6,6)):
        """ Initialises the measurement operators for u,v coordinates 'vis' (M x 2)"""
        self.op = KbNufftModule(im_size=Nd, grid_size=Kd, numpoints=Jd[0], norm='ortho')
        self.vis = tf.convert_to_tensor(vis)[None, ...]
        # self.op.batch = Nd[0]
        # self.op.plan(vis, Nd, Kd, Jd)
        # self.op.preapre_for_tf()
    
    @tf.function()
    def dir_op(self, x):
        # x = tf.cast(tf.convert_to_tensor(x), tf.complex64)[None, None, ...]
        y =  kbnufft_forward(self.op._extract_nufft_interpob())(x, self.vis)
        return y
    
    @tf.function()
    def adj_op(self, y):
        interpob = self.op._extract_nufft_interpob()
        nufft_adj = kbnufft_adjoint(interpob)
        x = nufft_adj(y, self.vis)        
        return x
    
    @tf.function()
    def self_adj(self, x):
        y = self.dir_op(x)
        x = self.adj_op(y)
        return x


class FFT_op():
    """Simple measurement operator. This operator maps the u,v coordinates to the nearest gridpoint. """
    def __init__(self, vis, Nd=(256,256), Kd=(512,512)):
        """ Initialises the measurement operators for u,v coordinates 'vis' (M x 2)"""

        self.Nd = Nd
        self.Kd = Kd
        self.vis = np.zeros(self.Kd, dtype=bool) #visability in 2d
        for u,v in vis:
            self.vis[ int(u/np.pi*Kd[0]//2 -.01)+Kd[0]//2, int(v/np.pi*Kd[1]//2-.01) + Kd[1]//2] = 1
        
    def dir_op(self, x):
        # zero pad in real space for upsampled k-space
        y = self._fft(np.pad(x, (
            ( (self.Kd[0]-self.Nd[0])//2, (self.Kd[0]-self.Nd[0])//2),
            ( (self.Kd[1]-self.Nd[1])//2, (self.Kd[1]-self.Nd[1])//2)
            )))  # measure only the given visabilities
        return y[self.vis]
    
    def adj_op(self, y):
        y_ = np.zeros(self.Kd, dtype=complex)
        y_[self.vis] = y
        x = self._ifft(y_) # use only given visabilities for reconstruction
        x = x[
            (self.Kd[0]-self.Nd[0])//2: (self.Kd[0]-self.Nd[0])//2 +self.Nd[0],
            (self.Kd[1]-self.Nd[1])//2: (self.Kd[1]-self.Nd[1])//2 +self.Nd[1]
            ] # remove zero padding from image
        return x
    
    def _fft(self, x):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x), norm='ortho'))

    def _ifft(self, Fx):
        return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Fx), norm='ortho')))

    def self_adj(self, x):
        y = self.dir_op(x)
        x = self.adj_op(y)
        return x