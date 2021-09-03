
import numpy as np
from pynufft import NUFFT

from tfkbnufft import kbnufft_forward, kbnufft_adjoint
from tfkbnufft.kbnufft import KbNufftModule


from scipy.special import iv, jv
# from scipy.sparse import *
import sparse
import tensorflow as tf
import tqdm


def calculate_kaiser_bessel_coef(k, i, Jd=(6,6)):
    """Calculate the Kaiser-Bessel kernel coefficients for a 2d grid for the neighbouring pixels. 

    Args:
        k (float,float): location of the point to be interpolated
        i (int): extra index parameter
        Jd (tuple, optional): Amount of neibhouring pixels to be used in each direction. Defaults to (6,6).

    Returns:
        indices (list): list of indices of all the calculated coefficients
        values (list): list of the calculated coefficients
    """

    k = k.reshape(-1,1)
    J = Jd[0]//2
    a = np.array(np.meshgrid(range(-J, J), range(-J, J))).reshape(2, -1)
    a += (k % 1 >0.5) # corrects to the closest 6 pixels
    indices = (k.astype(int) + a)

    J = Jd[0]

    beta = 2.34*J
    norm = J 

    # for 2d do the interpolation 2 times, once in each direction
    u =  k.reshape(2,1) - indices
    values1 = iv(0, beta*np.sqrt(1 +0j - (2*u[0]/Jd[0])**2)).real / J 
    values2 = iv(0, beta*np.sqrt(1 +0j - (2*u[1]/Jd[0])**2)).real / J 
    values = values1 * values2
    
    indices = np.vstack((
            np.zeros(indices.shape[1]), 
            np.repeat(i, indices.shape[1]), indices[0], indices[1])
            ).astype(int)

    return indices.T, values


class NUFFT_op():
    """NUFFT implementation using a Kaiser-Bessel kernel for interpolation. 
    """
    def __init__(self):
        pass
        # TODO generalise more, (pick axes, norm, etc.)
        
    def plan(self, uv, Nd, Kd, Jd, batch_size=None):
        # saving some values
        self.Nd = Nd
        self.Kd = Kd
        self.Jd = Jd
        self.M = len(uv)
        
        gridsize = 2*np.pi / Kd[0]
        k = (uv + np.pi) / gridsize
        
        # calculating coefficients for interpolation
        indices = []
        values =  []
        for i in tqdm.tqdm(range(len(uv))):
            ind, vals = calculate_kaiser_bessel_coef(k[i], i, Jd)
            indices.append(ind)
            values.append(vals.real)
        
        # building sparse matrix
        indices = np.array(indices).reshape(-1, 4)
        values = np.array(values).reshape(-1)
        self.interp_matrix = sparse.COO(indices.T, values, shape=(1, len(uv), Kd[0], Kd[1]))
    
        # calculating scaling based on iFT of the KB kernel
        J = Jd[0] 
        beta = 2.34*J
        s_kb = lambda x: np.sinc(np.sqrt((np.pi *x *J)**2 - (2.34*J)**2 +0j)/np.pi)

        # scaling done for both axes seperately
        xx = (np.arange(Kd[0])/Kd[0] -.5)[Kd[0]//4:-Kd[0]//4]
        sa = s_kb(xx).real
        self.scaling = (sa.reshape(-1,1) * sa.reshape(1,-1))

    def dir_op(self, xx):
        return np.squeeze(self._kk2k(self._xx2kk(self._pad((xx/self.scaling).reshape(-1, self.Nd[0], self.Nd[1])))))  # why divide on both sides?

    
    def adj_op(self, k):
        kk = self._k2kk(k)
        xx = self._kk2xx(kk)
        xx = self._unpad(xx)
        xx = np.squeeze(xx) / self.scaling
        return xx 
        # return np.squeeze(self._unpad(self._kk2xx(self._k2kk(k)))) / self.scaling
    
    def _kk2k(self, kk):
        """interpolates of the grid to non uniform measurements"""
        return (self.interp_matrix * kk.reshape(-1, 1, self.Kd[0], self.Kd[1])).sum(axis=(2,3)).todense()
    
    def _k2kk(self, y):
        """convolutes measurements to oversampled fft grid"""
        return (self.interp_matrix * y.reshape(-1, self.M, 1, 1)).sum(axis=1).todense()
    
    @staticmethod
    def _kk2xx(kk):
        """from 2d fourier space to 2d image space"""
        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kk, axes=(-2,-1)), axes=(-2,-1), norm=None), axes=(-2,-1))

    @staticmethod
    def _xx2kk(xx):
        """from 2d fourier space to 2d image space"""
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(xx, axes=(-2,-1)), axes=(-2,-1), norm=None), axes=(-2,-1))
    
    def _pad(self, x):
        """pads x to go from Nd to Kd"""
        return np.pad(x, (
            ( 0,0 ),
            ( (self.Kd[0]-self.Nd[0])//2, (self.Kd[0]-self.Nd[0])//2),
            ( (self.Kd[1]-self.Nd[1])//2, (self.Kd[1]-self.Nd[1])//2)
            ))
    
    
    def _unpad(self, x):
        """unpads x to go from  Kd to Nd"""
        return x[
            :,
            (self.Kd[0]-self.Nd[0])//2: (self.Kd[0]-self.Nd[0])//2 +self.Nd[0],
            (self.Kd[1]-self.Nd[1])//2: (self.Kd[1]-self.Nd[1])//2 +self.Nd[1]
            ] # remove zero padding from image



class NUFFT_op_TF():
    """NUFFT implementation using a Kaiser-Bessel kernel for interpolation. 
    Implemented with TF operations. Only able to do the FFT on the last 2 axes 
    of the tensors provided. Slower than using the numpy_function on the np 
    based operations.
    """
    def __init__(self):
        pass
        
    def plan(self, uv, Nd, Kd, Jd, batch_size):
        # saving some values
        self.Nd = Nd
        self.Kd = Kd
        self.Jd = Jd
        self.M = len(uv)
        
        gridsize = 2*np.pi / Kd[0]
        k = (uv + np.pi) / gridsize
        
        # calculating coefficients for interpolation
        indices = []
        values =  []
        for i in tqdm.tqdm(range(len(uv))):
            ind, vals = calculate_kaiser_bessel_coef(k[i], i, Jd)
            indices.append(ind)
            values.append(vals.real)
        
        # repeating the values and indices to match the batch_size (con of sparse tensors)
        values = np.array(values).reshape(-1)
        indices = np.array(indices).reshape(-1, 4)
        batch_indices = np.tile(indices[:,1:], [batch_size, 1])
        batch_indicators = np.repeat(np.arange(batch_size), (len(values)))
        batch_indices = np.hstack((batch_indicators[:,None], batch_indices))

        values = np.array(values).reshape(-1)
        batch_values = np.tile(values, batch_size).astype(np.float32)

        # build sparse matrix
        self.interp_matrix = tf.sparse.SparseTensor(batch_indices, batch_values, [batch_size, len(uv), Kd[0], Kd[1]])
        # self.interp_matrix = tf.sparse.reorder(self.interp_matrix)

        # determin scaling based on iFT of the KB kernel
        J = Jd[0] 
        beta = 2.34*J
        s_kb = lambda x: np.sinc(np.sqrt((np.pi *x *J)**2 - (2.34*J)**2 +0j)/np.pi)

        xx = (np.arange(Kd[0])/Kd[0] -.5)[Kd[0]//4:-Kd[0]//4]
        sa = s_kb(xx).real
        self.scaling = (sa.reshape(-1,1) * sa.reshape(1,-1)).reshape(1, Nd[0], Nd[0])
        self.forward = self.dir_op
        self.adjoint = self.adj_op

    def dir_op(self, xx):
        xx = tf.cast(xx, tf.complex64)
        xx = xx/self.scaling
        xx = self._pad(xx)
        
        kk = self._xx2kk(xx)
        
        kk = kk[:, None, :, :] # adding axes for sparse multiplication; shape [batch_size, 1, K, K]
        # split real and imaginary parts because complex operations not defined for sparseTensors
        k_real = tf.cast(self._kk2k(tf.math.real(kk)), tf.complex64)
        k_imag = tf.cast(self._kk2k(tf.math.imag(kk)), tf.complex64)
        return k_real + 1j * k_imag
        
    
    def adj_op(self, k):
        # split real and imaginary parts because complex operations not defined for sparseTensors
        k = k[:,:, None, None] # adding axes for sparse multiplication; shape [batch_size, M, 1, 1]
        k_real = tf.math.real(k)
        k_imag = tf.math.imag(k)
        kk_real = tf.cast(self._k2kk(k_real), tf.complex64)
        kk_imag = tf.cast(self._k2kk(k_imag), tf.complex64)
        kk = kk_real + 1j* kk_imag
        xx = self._kk2xx(kk)
        xx = self._unpad(xx)
        xx = xx / self.scaling
        return xx
    
    def learned_adj_op(self, k):
        """adjoint operation with a convolutional layer between interpolation and FT"""
        # split real and imaginary parts because complex operations not defined for sparseTensors
        k = k[:,:, None, None] # adding axes for sparse multiplication; shape [batch_size, M, 1, 1]
    
        k_real = tf.math.real(k)
        k_imag = tf.math.imag(k)
        # kk_real = self._k2kk(k_real)[:,:,:,None]
        # kk_imag = self._k2kk(k_imag)[:,:,:,None]
        
        kk_real = tf.expand_dims(self._k2kk(k_real), axis=3)
        kk_imag = tf.expand_dims(self._k2kk(k_imag), axis=3)

        conv = tf.keras.layers.Conv2D(1, (3,3), activation='relu', padding="same")
        
        # kk_real = tf.cast(conv(kk_real), tf.complex64)[:,:,:,0]
        # kk_imag = tf.cast(conv(kk_imag), tf.complex64)[:,:,:,0]
        
        kk_real = tf.squeeze(tf.cast(conv(kk_real), tf.complex64))
        kk_imag = tf.squeeze(tf.cast(conv(kk_imag), tf.complex64))

        kk = kk_real + 1j* kk_imag
        xx = self._kk2xx(kk)
        xx = self._unpad(xx)
        xx = xx / self.scaling
        return xx
    
    def _kk2k(self, kk):
        """interpolates of the grid to non uniform measurements"""
        return tf.sparse.reduce_sum(self.interp_matrix * kk, axis=(2,3))
            
    def _k2kk(self, k):
        """convolutes measurements to oversampled fft grid"""
        return tf.sparse.reduce_sum(self.interp_matrix * k, axis=1 )
    
    @staticmethod
    def _kk2xx(kk):
        """from 2d fourier space to 2d image space"""
        return tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.ifftshift(kk, axes=(-2,-1))), axes=(-2,-1))

    @staticmethod
    def _xx2kk(xx):
        """from 2d fourier space to 2d image space"""
        return tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(xx, axes=(-2,-1))), axes=(-2,-1))
    
    def _pad(self, x):
        """pads x to go from Nd to Kd"""
        return tf.pad(x, tf.constant([
            [0, 0],
            [(self.Kd[0]-self.Nd[0])//2, (self.Kd[0]-self.Nd[0])//2],
            [(self.Kd[1]-self.Nd[1])//2, (self.Kd[1]-self.Nd[1])//2]
        ]))
    
    
    def _unpad(self, x):
        """unpads x to go from  Kd to Nd"""
        return x[
            :,
            (self.Kd[0]-self.Nd[0])//2: (self.Kd[0]-self.Nd[0])//2 +self.Nd[0],
            (self.Kd[1]-self.Nd[1])//2: (self.Kd[1]-self.Nd[1])//2 +self.Nd[1]
            ] # remove zero padding from image


class old_NUFFT_op():
    """Simple measurement operator using the pyNUFFT package to sample at non-uniform u,v coordinates"""
    def __init__(self, vis, Nd=(256,256), Kd=(512,512), Jd=(6,6)):
        """ Initialises the measurement operators for u,v coordinates 'vis' (M x 2)"""
        self.op = NUFFT()
        self.op.plan(vis, Nd, Kd, Jd)
        self.n_measurements = len(vis)
        
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


class old_NUFFT_op_TF():
    """Simple measurement operator using the pyNUFFT package to sample at non-uniform u,v coordinates"""
    def __init__(self, vis, Nd=(256,256), Kd=(512,512), Jd=(6,6)):
        """ Initialises the measurement operators for u,v coordinates 'vis' (M x 2)"""
        self.op = KbNufftModule(im_size=Nd, grid_size=Kd, numpoints=Jd[0], norm='ortho')
        self.vis = tf.convert_to_tensor(vis)[None, ...]
        self.n_measurements = len(vis)

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
        self.n_measurements = len(vis)

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
    
