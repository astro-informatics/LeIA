import numpy as np


def RadialSampling(n_spokes=37, n_bins=10):
    """Returns the u,v coordinates for uniform sampling on radial spokes. Normalised to [-pi, pi]"""
    r   = np.linspace( np.pi, 0,  n_bins, endpoint=False)[::-1]
    phi = np.linspace(0, 2*np.pi, n_spokes,   endpoint=False)

    rr, pp = np.meshgrid(r, phi)
    
    vis = np.array([(rr*np.sin(pp)).reshape(-1), (rr*np.cos(pp)).reshape(-1)]).T
    return vis/np.max(vis) *np.pi


def SpiderSampling(n_spokes=37, n_lenslets=24, spectral_bins=10, max_baseline=0.65, min_baseline=0.15):
    """Returns the u,v coordinates for sampling following the SPIDER instrument. Normalised to [-pi, pi] (as described in Liu et al. 2018) """

    coordinates = np.array([ ( (lens+1)/n_lenslets * (max_baseline-min_baseline) + min_baseline)* np.array([np.cos(2*np.pi *spoke/n_spokes), np.sin(2*np.pi *spoke/n_spokes)]) for spoke in range(n_spokes) for lens in range(n_lenslets)])
    spectrals = np.linspace(500e-9, 900e-9, spectral_bins)

    vis = list()
    for n in range(n_spokes):
        for i in range(n_lenslets//2):
            x1 = coordinates[n*n_lenslets + i]
            x2 = coordinates[(n+1)*n_lenslets -i -1]
            for k in range(spectral_bins):
                vis.append(2*np.pi*(x1-x2)/spectrals[k])

    vis = np.array(vis)
    vis = vis / np.max( np.abs(vis)) *np.pi # normalise to [-pi, pi]
    return vis


def SpiderSampling2(n_spokes=37, n_lenslets=24, spectral_bins=10, max_baseline=0.65, min_baseline=0.15):
    """Returns the u,v coordinates for sampling following the SPIDER instrument. Normalised to [-pi, pi] (as described in Duncan et al. 2001) """

    baselines = np.array([3.6, 7.2, 10.8, 14.4, 18.0, 21.6, 28.8, 36.0, 46.8, 61.2, 79.2, 104.4]) # predefined baseline lengths in mm (Duncan et al. 2001)
   
    spectral_bins = np.linspace(500e-9, 900e-9, spectral_bins) /1e3 #mm

    vis = list()
    for n in range(n_spokes):
        angle = n/n_spokes * np.pi *2
        for i in range(len(baselines)):
            for k in range(len(spectral_bins)):
                l = baselines[i]/spectral_bins[k]
                vis.append((l*np.sin(angle), l*np.cos(angle))) 
    
    vis = np.array(vis)
    vis = vis / np.max( np.abs(vis)) *np.pi # normalise to [-pi, pi]
    return vis