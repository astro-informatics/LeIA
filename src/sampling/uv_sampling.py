import numpy as np


def radial_sampling(n_spokes=37, n_bins=10):
    """Returns the u,v coordinates for uniform sampling on radial spokes. Normalised to [-pi, pi]"""
    r   = np.linspace( np.pi, 0,  n_bins, endpoint=False)[::-1]
    phi = np.linspace(0, 2*np.pi, n_spokes,   endpoint=False)

    rr, pp = np.meshgrid(r, phi)
    
    vis = np.array([(rr*np.sin(pp)).reshape(-1), (rr*np.cos(pp)).reshape(-1)]).T
    return vis/np.max(vis) * np.pi /2


def spider_sampling(n_spokes=37, n_lenslets=24, n_wavelengths=10, normalised=True):
    angles = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)
    wavelengths = np.linspace(500e-9, 900e-9, n_wavelengths) 
    baselines = np.arange(1, n_lenslets, 2) # scaling doesn't matter, relative length of baseline
#     baselines = np.linspace(0.5, 0, n_lenslets//2, endpoint=False)[::-1] # scaling doesn't matter, relative length of baseline

    uv = list()
    for angle in angles: 
        for x in baselines:
            for k in wavelengths:
                    uv.append((x/k*np.sin(angle), x/k*np.cos(angle))) 
                    # uv.append((-x/k*np.sin(angle), -x/k*np.cos(angle))) 
    
    # normalise to -pi/2, pi/2
    uv = np.array(uv)
    if normalised:
        uv = uv / np.max( np.abs(uv)) * np.pi /2
    return uv
