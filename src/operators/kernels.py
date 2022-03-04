import numpy as np
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

