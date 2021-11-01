import os
import numpy as np

from astropy.io import fits 
from scipy.ndimage import gaussian_filter

from src.dataset import \
    measurement_func, random_crop, center_crop, \
    Dataset, EllipseDataset, make_yogadl_dataset

def load_M51():
    x_true = gaussian_filter(fits.getdata("./data/M51.fits"), 1)
    return x_true

def create_dataset(data, ISNR, size=2000):
    tf_func, func = measurement_func(ISNR=ISNR)
    ds = Dataset(size, data)
    yogadl_dataset = make_yogadl_dataset(ds, shuffle=False)

    if data in ["COCO", "SATS"] :
        dataset = ds.map(random_crop).map(tf_func) # crop randomly
    elif data == "GZOO":
        dataset = ds.map(center_crop).map(tf_func) # crop centre 
    elif data == "LLPS":
        ds = EllipseDataset(size) # randomly generated set
        yogadl_dataset = make_yogadl_dataset(ds, shuffle=False)
        dataset = ds.map(tf_func)
    return dataset

def create_train_test(data, ISNR):
    a = create_dataset(data, 30, 3000)
    dat = np.array([i for i in a])
    y_data = np.array([i[0] for i in dat])
    x_data = np.array([i[1] for i in dat])

    folder = f"./data/intermediate/{data}"
    if not os.path.exists(folder):
        os.mkdir(folder)
    np.save(f"{folder}/x_true_train_30dB.npy",  x_data[:2000])
    np.save(f"{folder}/x_true_test_30dB.npy",   x_data[2000:])
    np.save(f"{folder}/y_dirty_train_30dB.npy", y_data[:2000])
    np.save(f"{folder}/y_dirty_test_30dB.npy",  y_data[2000:])

def create_generalisation_set(data_list=[], ISNR=30):
    x_gen, y_gen = None, None
    for data in data_list:
        folder = f"./data/intermediate/{data}"

        y_dirty = np.squeeze(np.load(f"{folder}/y_dirty_test_30dB.npy")[:1000])
        x_dirty = np.squeeze(np.load(f"{folder}/x_true_test_30dB.npy")[:1000])
        if x_gen is None:
            x_gen = x_dirty
            y_gen = y_dirty
        else:
            x_gen = np.vstack((x_gen, x_dirty))
            y_gen = np.vstack((y_gen, y_dirty))

    np.save(f"./data/intermediate/y_dirty_gen_30dB.npy",  y_gen)
    np.save(f"./data/intermediate/x_true_gen_30dB.npy",  x_gen)


