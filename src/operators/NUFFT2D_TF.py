# import sparse
import tqdm
import numpy as np
import tensorflow as tf
from src.operators.kernels import calculate_kaiser_bessel_coef



class NUFFT2D_TF():
    """NUFFT implementation using a Kaiser-Bessel kernel for interpolation. 
    Implemented with TF operations. Only able to do the FFT on the last 2 axes 
    of the tensors provided. Slower than using the numpy_function on the np 
    based operations.
    """
    def __init__(self):
        pass
        
    def plan(self, uv, Nd, Kd, Jd, batch_size, measurement_weights=1, normalize=False):
        # saving some values
        self.Nd = Nd
        self.Kd = Kd
        self.Jd = Jd
        self.M = len(uv)
        self.measurement_weights = measurement_weights
        self.normalization_factor = 1
        self.batch_size = batch_size
        self.n_measurements = len(uv)
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
        
        #check if indices are within bounds, otherwise suppress them and raise warning
        if np.any(indices[:,2:] < 0) or np.any(indices[:,2:] >= Kd[0]):
            sel_out_bounds = (np.any(indices[:,2:] < 0, axis=1) | np.any(indices[:,2:] >= Kd[0], axis=1))
            print(f"some values lie out of the interpolation array, these are not used, check baselines")
            indices = indices[~sel_out_bounds]
            values = values[~sel_out_bounds]
        
        
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

        if normalize:
            self._normalize()

    def dir_op(self, xx):
        xx = tf.cast(xx, tf.complex64)
        xx = xx/self.scaling
        xx = self._pad(xx)
        
        kk = self._xx2kk(xx) / self.Kd[0]
        
        kk = kk[:, None, :, :] # adding axes for sparse multiplication; shape [batch_size, 1, K, K]
        # split real and imaginary parts because complex operations not defined for sparseTensors
        k_real = tf.cast(self._kk2k(tf.math.real(kk)), tf.complex64)
        k_imag = tf.cast(self._kk2k(tf.math.imag(kk)), tf.complex64)
        return k_real + 1j * k_imag
        
    
    def adj_op(self, k, measurement_weighting=False):
        # split real and imaginary parts because complex operations not defined for sparseTensors
        if measurement_weighting:
            k = k * self.measurement_weights # weighting measurements
        k = k[:,:, None, None] # adding axes for sparse multiplication; shape [batch_size, M, 1, 1]
        k_real = tf.math.real(k)
        k_imag = tf.math.imag(k)
        kk_real = tf.cast(self._k2kk(k_real), tf.complex64)
        kk_imag = tf.cast(self._k2kk(k_imag), tf.complex64)
        kk = kk_real + 1j* kk_imag
        xx = self._kk2xx(kk) * self.Kd[0]
        xx = self._unpad(xx)
        xx = xx / self.scaling
        if measurement_weighting:
            xx = xx / self.normalization_factor # normalising for operator norm
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

    def _normalize(self):
        # normalise operator norm on random (0,1) image
        t = np.random.normal(size=(self.batch_size, self.Nd[0], self.Nd[1])).astype(np.float32)
        t = (t - t.min())/(t.max()-t.min())
#         t = self.adj_op(self.dir_op(t))
        new_t = self.adj_op(self.dir_op(t), measurement_weighting=True)
        
        self.nu =  tf.norm(tf.math.real(new_t))/tf.norm(t)