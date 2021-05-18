from astropy.io import fits 
from scipy.ndimage import gaussian_filter

def load_M51():
    x_true = gaussian_filter(fits.getdata("./data/M51.fits"), 1)
    return x_true