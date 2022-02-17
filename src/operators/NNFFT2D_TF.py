import numpy as np
import tensorflow as tf
import tqdm

class NNFFT2D_TF():
    """Nearest Neighbour FFT implementation using TensorFlow. This operator maps the u,v coordinates to the nearest gridpoint and does an FFT.
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
            ind, vals = (0, i, int(k[i,0]+.5), int(k[i,1] + 0.5)), 1
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

    

    def dir_op(self, xx):
        xx = tf.cast(xx, tf.complex64)
        # xx = xx/self.scaling
        xx = self._pad(xx)
        
        kk = self._xx2kk(xx) / self.Kd[0] # unitary
        
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
        # xx = xx / self.scaling
        if measurement_weighting:
            xx = xx / self.normalization_factor # normalising for operator norm
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