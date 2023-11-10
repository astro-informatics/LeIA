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
        
        # repeating the values and indices to match the batch_size 
        values = np.array(values).reshape(-1)
        indices = np.array(indices).reshape(-1, 4)
        
        batch_indices = np.tile(indices[:,-2:], [batch_size, 1])
        batch_indicators = np.repeat(np.arange(batch_size), (len(values)))
        self.batch_indices = np.hstack((batch_indicators[:,None], batch_indices))

        values = np.array(values).reshape(-1)
        self.batch_values = np.tile(values, [batch_size,1]).astype(np.float32).reshape(self.batch_size, self.n_measurements, self.Jd[0]*self.Jd[1])

        # check if indices are within bounds, otherwise suppress them and raise warning
        if np.any(self.batch_indices[:,-2:] < 0) or np.any(self.batch_indices[:,-2:] >= Kd[0]):
            self.sel_out_bounds = (np.any(self.batch_indices[:,-2:] < 0, axis=1) | np.any(self.batch_indices[:,-2:] >= Kd[0], axis=1))
            print(f"some values lie out of the interpolation array, these are not used, check baselines")
            
            # Since we want to keep the shape of the interpolation array the same, 
            # we will relocate the out of bounds indices to (0,0), 
            # but set the interpolation coefficients to zero so they are not counted. 
            self.batch_indices[self.sel_out_bounds] = np.zeros(self.batch_indices.shape[1])
            self.batch_values[self.sel_out_bounds.reshape(self.batch_values.shape)] = 0

        # determine scaling based on iFT of the KB kernel
        J = Jd[0] 
        beta = 2.34*J
        s_kb = lambda x: np.sinc(np.sqrt((np.pi *x *J)**2 - (2.34*J)**2 +0j)/np.pi)

        xx = (np.arange(Kd[0])/Kd[0] -.5)[(Kd[0]-Nd[0])//2:(Kd[0]-Nd[0])//2 + Nd[0]]
        sa = s_kb(xx).real
        self.scaling = (sa.reshape(-1,1) * sa.reshape(1,-1)).reshape(1, Nd[0], Nd[0])
        self.scaling = tf.convert_to_tensor(self.scaling, dtype=tf.complex64)
        self.forward = self.dir_op
        self.adjoint = self.adj_op

        if normalize:
            self._normalize()

    def dir_op(self, xx):
        xx = tf.cast(xx, tf.complex64)
        xx = xx/self.scaling
        xx = self._pad(xx)
        kk = self._xx2kk(xx) / self.Kd[0]
        k = self._kk2k(kk)
        return k


    def adj_op(self, k, measurement_weighting=False):
        # split real and imaginary parts because complex operations not defined for sparseTensors
        if measurement_weighting:
            k = k * self.measurement_weights # weighting measurements
        kk = self._k2kk(k)
        xx = self._kk2xx(kk) * self.Kd[0]
        xx = self._unpad(xx)
        xx = xx / self.scaling
        if measurement_weighting:
            xx = xx / self.normalization_factor # normalising for operator norm
        return xx
    

    def _kk2k_sub(self, kk, sel):
        """interpolates of the grid to non uniform measurements"""
        batch_indices = self.batch_indices.reshape(self.batch_size, -1, self.Jd[0]*self.Jd[1], 3)
        sel_batch_indices = tf.reshape(tf.boolean_mask(batch_indices, sel, axis=1), [-1,3])
        sel_batch_values = tf.cast(tf.boolean_mask(self.batch_values, sel, axis=1), tf.complex64)
        
        v = tf.gather_nd(kk, sel_batch_indices)
        r = tf.reshape(v, (self.batch_size, -1, self.Jd[0]*self.Jd[1])) * sel_batch_values
        return tf.reduce_sum(r, axis=2)

    def dir_op_sub(self, xx, sel):
        xx = tf.cast(xx, tf.complex64)
        xx = xx/self.scaling
        xx = self._pad(xx)
        kk = self._xx2kk(xx) / self.Kd[0]
        k = self._kk2k_sub( kk, sel)
        return k

    def _k2kk_sub(self, k_sub, sel):
        """convolutes measurements to oversampled fft grid"""
#         interp = k_sub[:,:,None] * self.batch_values[:, sel]
        interp = k_sub[:,:,None] * tf.cast(tf.boolean_mask(self.batch_values, sel, axis=1), tf.complex64)
        interp = tf.reshape(interp, [-1])

        batch_indices = self.batch_indices.reshape(self.batch_size, -1, self.Jd[0]*self.Jd[1], 3)
        sel_batch_indices = tf.reshape(tf.boolean_mask(batch_indices, sel, axis=1), [-1,3])

        f = tf.scatter_nd(sel_batch_indices, interp, [self.batch_size] + list(self.Kd))
        return f

    def adj_op_sub(self, k, sel, measurement_weighting=False):
        # split real and imaginary parts because complex operations not defined for sparseTensors
        if measurement_weighting:
            k = k * self.measurement_weights # weighting measurements
        kk = self._k2kk_sub( k, sel)
        xx = self._kk2xx(kk) * self.Kd[0]
        xx = self._unpad(xx)
        xx = xx / self.scaling
        if measurement_weighting:
            xx = xx / self.normalization_factor # normalising for operator norm
        return xx

    def _kk2k(self, kk):
        """interpolates of the grid to non uniform measurements"""
        v = tf.gather_nd(kk, self.batch_indices)
        r = tf.reshape(v, (self.batch_size, self.n_measurements, self.Jd[0]*self.Jd[1])) * self.batch_values
        return tf.reduce_sum(r, axis=2)
        
    def _k2kk(self, k):
        """convolutes measurements to oversampled fft grid"""
        interp = k[:,:,None] * self.batch_values
        interp = tf.reshape(interp, [-1])

        f = tf.scatter_nd(self.batch_indices, interp, [self.batch_size] + list(self.Kd))
        return f
        
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
            ] 

    def _normalize(self):
        # normalise operator norm on random (0,1) image
        t = np.random.normal(size=(self.batch_size, self.Nd[0], self.Nd[1])).astype(np.float32)
        t = (t - t.min())/(t.max()-t.min())
#         t = self.adj_op(self.dir_op(t))
        new_t = self.adj_op(self.dir_op(t), measurement_weighting=True)
        
        self.nu =  tf.norm(tf.math.real(new_t))/tf.norm(t)