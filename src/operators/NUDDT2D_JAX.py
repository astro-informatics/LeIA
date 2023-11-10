import numpy as np
import jax
import jax.numpy as jnp

from scipy.special import iv, jv

def calculate_kaiser_bessel_coef(k, i, Jd=(6,6)):
    """Calculate the Kaiser-Bessel kernel coefficients for a 2d grid for the neighbouring pixels. 

    Args:
        k (float,float): location of the point to be interpolated
        i (int): extra index parameter
        Jd (tuple, optional): Amount of neighbouring pixels to be used in each direction. Defaults to (6,6).

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


def gather_nd_unbatched(params, indices):
  return params[tuple(jnp.moveaxis(indices, -1, 0))]

def gather_nd(params, indices, batch=False):
  if not batch:
    return gather_nd_unbatched(params, indices)
  else:
    return jax.vmap(gather_nd_unbatched, (0, 0), 0)(params, indices)

def scatter_nd(indices, updates, shape):
    zeros = jnp.zeros(shape, updates.dtype)
    key = tuple(jnp.moveaxis(indices, -1, 0))
    return zeros.at[key].add(updates)


class NUFFT2D_jax():
    """JAX implementation of the 2D Non-Uniform FFT using a Kaiser-Bessel kernel for interpolation. 
    """
    def __init__(self):
        pass
        
    def plan(self, uv, Nd, Kd, Jd, batch_size=None):
        self.Nd = Nd
        self.Kd = Kd
        self.Jd = Jd
        self.M = len(uv)
        self.normalization_factor = 1
        self.batch_size = batch_size
        self.n_measurements = len(uv)

        gridsize = 2*np.pi / Kd[0]
        k = (uv + np.pi) / gridsize
        
        # calculating coefficients and their indices for interpolation
        indices = []
        values =  []
        for i in tqdm.tqdm(range(len(uv))):
            ind, vals = calculate_kaiser_bessel_coef(k[i], i, Jd)
            indices.append(ind)
            values.append(vals.real)
        
        # repeating the values and indices to match the batch_size 
        values = np.array(values).reshape(-1)
        indices = np.array(indices).reshape(-1, 4)
        
        # indices and values are duplicated for every image in the batch_size. This is not memory efficient but is computationally easier. 
        batch_indices = np.tile(indices[:,-2:], [batch_size, 1])
        batch_indicators = np.repeat(np.arange(batch_size), (len(values)))
        self.batch_indices = np.hstack((batch_indicators[:,None], batch_indices))

        values = np.array(values).reshape(-1)
        self.batch_values = np.tile(values, [batch_size,1]).astype(np.float32).reshape(self.batch_size, self.n_measurements, self.Jd[0]*self.Jd[1])

        # check if indices are within bounds, otherwise suppress them and raise warning. (This happens when the kernel is situated close to the edge of the bounded area)
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

        xx = (np.arange(Kd[0])/Kd[0] -.5)[Kd[0]//4:-Kd[0]//4] #TODO remove hard coded upsampling of factor 4 in slicing
        sa = s_kb(xx).real
        self.scaling = (sa.reshape(-1,1) * sa.reshape(1,-1)).reshape(1, Nd[0], Nd[0])

        # creating some aliases
        self.forward = self.dir_op
        self.adjoint = self.adj_op


    def dir_op(self, xx):
        """Forward operation mapping from a set of 2d images to non-uniformly distributed measurements"""
        if xx.ndim == 2:
            xx = xx[np.newaxis, :]
        return np.squeeze(self._kk2k(self._xx2kk(self._pad((xx/self.scaling).reshape(-1, self.Nd[0], self.Nd[1])))))  

    def adj_op(self, k):
        """Adjoint operation mapping from non-uniformly distributed measurements to an image"""
        if k.ndim == 1:
            k = k[np.newaxis, :]
        kk = self._k2kk(k)
        xx = self._kk2xx(kk)
        xx = self._unpad(xx)
        xx = np.squeeze(xx) / self.scaling
        return xx 


    def _kk2k(self, kk):
        """interpolates of the grid to non uniform measurements"""
        v = gather_nd(kk, self.batch_indices) # gather values from Fourier image
        r = jax.lax.reshape(v, (self.batch_size, self.n_measurements, self.Jd[0]*self.Jd[1])) * self.batch_values # reshape and multiply with interpolation coefficients
        return jnp.sum(r, axis=2) # sum the contributions from all the measurements into the images
    
    def _k2kk(self, k):
        """convolves measurements to oversampled fft grid"""
        interp = k[:,:,None] * self.batch_values # multiply with interpolation coefficients
        interp = jax.lax.reshape(interp, [jnp.size(interp)]) # flatten

        f = scatter_nd(self.batch_indices, interp, [self.batch_size] + list(self.Kd)) # scatter interpolated values onto an empty batch of Fourier images
        return f 
    
    @staticmethod
    def _kk2xx(kk):
        """from 2d fourier space to 2d image space"""
        return jnp.fft.ifftshift(jnp.fft.ifft2(jnp.fft.ifftshift(kk, axes=(-2,-1)), axes=(-2,-1), norm='ortho'), axes=(-2,-1))

    @staticmethod
    def _xx2kk(xx):
        """from 2d fourier space to 2d image space"""
        return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(xx, axes=(-2,-1)), axes=(-2,-1), norm='ortho'), axes=(-2,-1))
    
    def _pad(self, x):
        """pads x to go from Nd to Kd"""
        return jnp.pad(x, (
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