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
#     data = "GZOO"
    tf_func, func = measurement_func(ISNR=ISNR)
    ds = Dataset(size, data)
    yogadl_dataset = make_yogadl_dataset(ds, shuffle=False)

    if data == "COCO":
        dataset = ds.map(random_crop).map(tf_func)#.map(data_map)
    elif data == "GZOO":
        dataset = ds.map(center_crop).map(tf_func)#.map(data_map)
    elif data == "LLPS":
        ds = EllipseDataset(size)
        yogadl_dataset = make_yogadl_dataset(ds, shuffle=False)
        dataset = ds.map(tf_func)
    return dataset

def create_train_test(data, ISNR):
    data = "LLPS"
    a = create_dataset(data, 30, 3000)
    dat = np.array([i for i in a])
    y_data = np.array([i[0] for i in dat])
    x_data = np.array([i[1] for i in dat])

    np.save(f"./data/intermediate/{data}/x_true_train_30dB.npy", x_data[:2000])
    np.save(f"./data/intermediate/{data}/x_true_test_30dB.npy", x_data[2000:])
    np.save(f"./data/intermediate/{data}/y_dirty_train_30dB.npy", y_data[:2000])
    np.save(f"./data/intermediate/{data}/y_dirty_test_30dB.npy", y_data[2000:])