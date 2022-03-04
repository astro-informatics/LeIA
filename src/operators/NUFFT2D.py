import sparse
import tqdm
import numpy as np
from src.operators.kernels import calculate_kaiser_bessel_coef



class NUFFT2D():
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
        
        # building sparse matrix
        indices = np.array(indices).reshape(-1, 4)
        values = np.array(values).reshape(-1)
        
        #check if indices are within bounds, otherwise suppress them and raise warning
        # TODO check in both dimensions instead of assuming square
        if np.any(indices[:,2:] < 0) or np.any(indices[:,2:] >= Kd[0]):
            sel_out_bounds = (np.any(indices[:,2:] < 0, axis=1) | np.any(indices[:,2:] >= Kd[0], axis=1))
            print(f"some values lie out of the interpolation array, these are not used, check baselines")
            indices = indices[~sel_out_bounds]
            values = values[~sel_out_bounds]
            
        
        self.interp_matrix = sparse.COO(indices.T, values, shape=(1, len(uv), Kd[0], Kd[1]))
    
        # calculating scaling based on iFT of the KB kernel
        J = Jd[0] 
        beta = 2.34*J
        s_kb = lambda x: np.sinc(np.sqrt((np.pi *x *J)**2 - (2.34*J)**2 +0j)/np.pi)

        # scaling done for both axes seperately
        xx = (np.arange(Kd[0])/Kd[0] -.5)[Kd[0]//4:-Kd[0]//4]
        sa = s_kb(xx)
        self.scaling = (sa.reshape(-1,1) * sa.reshape(1,-1))

    def dir_op(self, xx):
        if xx.ndim == 2:
            xx = xx[np.newaxis, :]
        return np.squeeze(self._kk2k(self._xx2kk(self._pad((xx/self.scaling).reshape(-1, self.Nd[0], self.Nd[1])))))  # why divide on both sides?

    def adj_op(self, k):
        if k.ndim == 1:
            k = k[np.newaxis, :]
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
        return (self.interp_matrix * y.reshape(-1, self.n_measurements, 1, 1)).sum(axis=1).todense()
    
    @staticmethod
    def _kk2xx(kk):
        """from 2d fourier space to 2d image space"""
        return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(kk, axes=(-2,-1)), axes=(-2,-1), norm='ortho'), axes=(-2,-1))

    @staticmethod
    def _xx2kk(xx):
        """from 2d fourier space to 2d image space"""
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(xx, axes=(-2,-1)), axes=(-2,-1), norm='ortho'), axes=(-2,-1))
    
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