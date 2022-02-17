import numpy as np

class NNFFT2D():
    """Nearest Neighbour FFT. This operator maps the u,v coordinates to the nearest gridpoint and does an FFT. """
    def plan(self, vis, Nd=(256,256), Kd=(512,512), Jd=None):
        """ Initialises the measurement operators for u,v coordinates 'vis' (M x 2)"""

        self.Nd = Nd
        self.Kd = Kd
        self.vis = np.zeros(self.Kd, dtype=bool) #visability in 2d

        for u,v in vis:
            self.vis[ int(u/np.pi*Kd[0]//2 -.01)+Kd[0]//2, int(v/np.pi*Kd[1]//2-.01) + Kd[1]//2] = 1
        
        self.n_measurements = int(np.sum(vis))

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
    
    @staticmethod
    def _fft(x):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x), norm='ortho'))

    @staticmethod
    def _ifft(Fx):
        return np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(Fx), norm='ortho')))

    def self_adj(self, x):
        y = self.dir_op(x)
        x = self.adj_op(y)
        return x