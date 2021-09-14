import glob
import numpy as np

from skimage import io, color
from tqdm import tqdm
import os

from src.operators.measurement import NUFFT_op
from src.sampling.uv_sampling import spider_sampling


def measure(x, m_op, ISNR, weights, factor=1):
    y0 = m_op.dir_op(x)
    sigma = np.sqrt(np.mean(np.abs(y0)**2)) * 10**(-ISNR/20)
    noise = np.random.normal(0, sigma, y0.shape) + 1j * np.random.normal(0, sigma, y0.shape)
    y = y0 + factor * noise
    x_filtered = m_op.adj_op(y*weights)
    noise_val = np.sqrt(2)*sigma #np.std((m_op.adj_op(noise)))
    return y, x_filtered.real, noise_val


def create_dataset(
    m_op,
    measurement_weights=1,
    ISNR=[50],
    image_size=256,
    directory = "./data/BSR/BSDS500/data/images/train/", 
    extension='.jpg',
    save_postfix="",
    start_file=0,
    n_files=None
    ):

    #    if type(ISNR) != list:
    #        ISNR = [ISNR]
    # files = glob.glob(directory + "*" + extension)
    files = np.loadtxt(os.environ["HOME"] + 
            "/src_aiai/images.txt", dtype=str)
    if n_files is None:
        n_files = len(files)
    n_images = n_files * len(ISNR)

    x_true  = np.zeros((n_images, image_size, image_size))
    x_dirty = np.zeros((n_images, image_size, image_size))
    y_dirty = np.zeros((n_images, m_op.M), dtype=complex)
    noise_levels = np.zeros((n_images))

    for i in tqdm(range(n_files), desc="Processing images: "):
        file_name = files[start_file + i]
        im = io.imread(file_name)
        start_pix = (im.shape[0]//2-image_size//2, im.shape[1]//2-image_size//2)
        im = color.rgb2gray(im[start_pix[0]:start_pix[0]+image_size,
            start_pix[1]:start_pix[1]+image_size])

        for j in range(len(ISNR)):
            y, x_filtered, noise_val = measure(im, m_op, ISNR[j], measurement_weights)

            x_true[i + j*n_files] = im
            x_dirty[i + j*n_files] = x_filtered
            y_dirty[i + j*n_files] = y
            noise_levels[i + j*n_files] = noise_val

    np.save("./data/intermediate/x_true_" + save_postfix +".npy", x_true)
    np.save("./data/intermediate/x_dirty_" + save_postfix +".npy", x_dirty)
    np.save("./data/intermediate/y_dirty_" + save_postfix +".npy", y_dirty)
    np.save("./data/intermediate/noise_levels_" + save_postfix +".npy", noise_levels)


uv = spider_sampling()
m_op = NUFFT_op()
m_op.plan(uv, (256,256), (512,512), (6,6))
dist = -1 * np.dot(uv, uv.T) -1 * np.dot(uv, uv.T).T + np.sum(uv**2, axis=1) + np.sum(uv**2, axis=1)[:,np.newaxis] # (x-y)'(x-y) = x'x + y'y - x'y -y'x
dist[dist < 0] = 0 # correct for numerical errors
gridsize = 2*np.pi/512
w = 1/np.sum(dist**.5 < gridsize, axis=1) # all pixels within 1 gridcell distancex_f = m_op.adj_op(y_*w/w.max())

# create_dataset(m_op, measurement_weights=w, ISNR=50, 
#     directory="./data/BSR/BSDS500/data/images/train/",
#     save_postfix="train_50dB")

# create_dataset(m_op, measurement_weights=w, ISNR=30, 
#     directory="./data/val2017/",
#     save_postfix="train_30dB")

# create_dataset(m_op, measurement_weights=w, ISNR=50, 
#     directory="./data/BSR/BSDS500/data/images/test/",
#     save_postfix="test_50dB")

# create_dataset(m_op, measurement_weights=w, ISNR=30, 
#     directory="./data/BSR/BSDS500/data/images/test/",
#     save_postfix="test_30dB")

create_dataset(m_op, measurement_weights=w, ISNR=np.arange(30,10,-2.5), 
    directory="./data/val2017/",
    save_postfix="test_robustness",
    start_file=3000,
    n_files=200)
