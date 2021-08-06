import glob
import numpy as np

from skimage import io, color
from tqdm import tqdm

from src.operators.measurement import NUFFT_op
from src.sampling.uv_sampling import spider_sampling


def measure(x, m_op, ISNR, weights):
    y0 = m_op.dir_op(x)
    sigma = np.sqrt(np.mean(np.abs(y0)**2)) * 10**(-ISNR/20)
    noise = np.random.normal(0, sigma, y0.shape) + 1j * np.random.normal(0, sigma, y0.shape)
    y = y0 + noise
    x_filtered = m_op.adj_op(y*weights)
    noise_val = np.sqrt(2)*sigma #np.std((m_op.adj_op(noise)))
    return y, x_filtered.real, noise_val


def create_dataset(
    m_op,
    measurement_weights=1,
    ISNR=50,
    image_size=256,
    directory = "./data/BSR/BSDS500/data/images/train/", 
    extension='.jpg',
    save_postfix=""
    ):

    files = glob.glob(directory + "*" + extension)
    x_true  = np.zeros((len(files), image_size, image_size, 1))
    x_dirty = np.zeros((len(files), image_size, image_size, 1))
    y_dirty = np.zeros((len(files), m_op.n_measurements, 1), dtype=complex)
    noise_levels = np.zeros((len(files),1))

    for i, file_name in tqdm(enumerate(files), desc="Processing images: "):
        im = io.imread(file_name)
        start_pix = (im.shape[0]//2-image_size//2, im.shape[1]//2-image_size//2)
        im = color.rgb2gray(im[start_pix[0]:start_pix[0]+image_size,
            start_pix[1]:start_pix[1]+image_size])


        y, x_filtered, noise_val = measure(im, m_op, ISNR, measurement_weights)

        x_true[i,:,:,0] = im
        x_dirty[i,:,:,0] = x_filtered
        y_dirty[i,:,0] = y
        noise_levels[i,0] = noise_val

    np.save("./data/intermediate/x_true_" + save_postfix +".npy", x_true)
    np.save("./data/intermediate/x_dirty_" + save_postfix +".npy", x_dirty)
    np.save("./data/intermediate/y_dirty_" + save_postfix +".npy", y_dirty)
    np.save("./data/intermediate/noise_levels_" + save_postfix +".npy", noise_levels)


uv = spider_sampling()
m_op = NUFFT_op(uv)
dist = -1 * np.dot(uv, uv.T) -1 * np.dot(uv, uv.T).T + np.sum(uv**2, axis=1) + np.sum(uv**2, axis=1)[:,np.newaxis] # (x-y)'(x-y) = x'x + y'y - x'y -y'x
dist[dist < 0] = 0 # correct for numerical errors
gridsize = 2*np.pi/512
w = 1/np.sum(dist**.5 < gridsize, axis=1) # all pixels within 1 gridcell distancex_f = m_op.adj_op(y_*w/w.max())

# create_dataset(m_op, measurement_weights=w, ISNR=50, 
#     directory="./data/BSR/BSDS500/data/images/train/",
#     save_postfix="train_50dB")

create_dataset(m_op, measurement_weights=w, ISNR=30, 
    directory="./data/val2017/",
    save_postfix="train_30dB")

# create_dataset(m_op, measurement_weights=w, ISNR=50, 
#     directory="./data/BSR/BSDS500/data/images/test/",
#     save_postfix="test_50dB")

create_dataset(m_op, measurement_weights=w, ISNR=30, 
    directory="./data/BSR/BSDS500/data/images/test/",
    save_postfix="test_30dB")