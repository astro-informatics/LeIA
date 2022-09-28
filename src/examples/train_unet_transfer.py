from functools import partial

import os
import sys 
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

tf.compat.v1.disable_eager_execution() # GUNet cannot use eager execution


# operators and sampling patterns
from src.operators.NUFFT2D import NUFFT2D
from src.operators.NUFFT2D_TF import NUFFT2D_TF
from src.sampling.uv_sampling import spider_sampling, random_sampling

 # some custom callbacks
from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 

# model and dataset generator
from src.networks.UNet import UNet
from src.networks.GUnet import GUnet
from src.networks.PseudoInverse import PseudoInverse


from src.dataset import Dataset, PregeneratedDataset, center_crop, data_map, make_yogadl_dataset, measurement_func, random_crop, data_map_image

# selecting one gpu to train on
from src.util import gpu_setup
gpu_setup()

import pandas as pd
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

from skimage import io, color
import glob
# load new data
# train model on purely data
# apply pre-trained model on data
# apply transfer-learned model to data

def predict(x_true, y_dirty, x_true_test, y_dirty_test, model, batch_size, data, ISNR, postfix=""):
    print("predict train")
    train_predict = model.predict(y_dirty, batch_size=batch_size)
    print("predict test")
    test_predict = model.predict(y_dirty_test, batch_size=batch_size)

    print("saving train and test predictions")
    os.makedirs(f"./data/processed/{data}/{operator}", exist_ok=True)
    np.save(f"./data/processed/{data}/{operator}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
    np.save(f"./data/processed/{data}/{operator}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)


    metrics = [
        ("PSNR", peak_signal_noise_ratio),
        ("SSIM", structural_similarity),
        ("MSE", mean_squared_error)
    ]

    statistics = pd.DataFrame(columns=["PSNR", "SSIM", "MSE", "method", "set"])
    name = f"{network} {operator} {postfix[1:]}"

    for dset, x, pred in [("train", x_true, train_predict), ("test", x_true_test, test_predict)]:
        df = pd.DataFrame()
        for metric, f in metrics:
            df[metric] = [f(x[i], pred[i]) for i in range(len(x))]
            df['Method'] = name
            df['Set'] = dset
            if statistics.empty:
                statistics = df
            else:
                statistics = statistics.append(df, ignore_index=False)

    print("saving results")
    with pd.option_context('mode.use_inf_as_na', True):
        statistics.dropna(inplace=True)

    os.makedirs(f"./results/{data}/{operator}/", exist_ok=True)
    statistics.to_csv(f"./results/{data}/{operator}/statistics_{network}_{ISNR}dB{postfix}.csv")
#TODO add a nice argument parser

epochs = 100
set_size = 2000 # size of the train set
save_freq = 20 # save every 20 epochs
batch_size = 20 
max_train_time = 40*60 # time after which training should stop in mins


ISNR = 30 #dB
network = "GUnet"
net = GUnet
activation = "linear"


try: 
    postfix = "_" + str(sys.argv[3])
except:
    postfix = ""


data = "SATS"

# creating the operator
Nd = (256, 256)
Kd = (512, 512)
Jd = (6,6)

operator = "NUFFT_SPIDER"
uv = spider_sampling()
y_shape = len(uv)
op = NUFFT2D_TF

# sampling density based weighting
grid_cell = 2*np.pi /512 
binned = (uv[:,:]+np.pi+.5*grid_cell) // grid_cell
binned = [tuple(x) for x in binned]
cells = set(binned)
w_gridded = np.zeros(uv.shape[0])
for cell in list(cells):
    mask = np.all(np.array(cell) ==  binned, axis=1)
    w_gridded[mask] = np.sum(mask)

# w = 
w = 1/w_gridded
w /= w.max()

# create dataset
class TNGDataset(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator():

        x = np.load(f"./data/intermediate/TNG/TNG50-1_halpha_484files.npy")
        while True:
            yield x[:,:,:,np.newaxis]

    def __new__(cls):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float32),
            args=()
        )

class SATSDataset(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator():
        files = glob.glob(os.environ["HOME"] +"/src_aiai/data/sat/train/*.jpg")  
        n = len(files) 
        i = 0  
        while True:
            x = io.imread(files[i], as_gray=True)
            yield x[:,:,np.newaxis]
            i = (i + 1) % n

    def __new__(cls):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float32),
            args=()
        )

# use only the first 300 for training
if data == "TNG":
    tf_func, func = measurement_func(uv,  m_op=None, Nd=(256,256), data_shape=y_shape, ISNR=ISNR)
    train_size = 300
    set_size = train_size
    ds = TNGDataset().unbatch().take(train_size).cache()
    # yogadl_dataset = make_yogadl_dataset(ds) # for efficient shuffling
    dataset = ds.repeat(epochs).shuffle(set_size).map(center_crop).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)

    # creating train and test set 
    try:
        x_train = np.load(f"./data/intermediate/TNG/{operator}/x_true_train_{ISNR}dB.npy")
        y_train= np.load(f"./data/intermediate/TNG/{operator}/y_dirty_train_{ISNR}dB.npy")

        x_test = np.load(f"./data/intermediate/TNG/{operator}/x_true_test_{ISNR}dB.npy")
        y_test = np.load(f"./data/intermediate/TNG/{operator}/y_dirty_test_{ISNR}dB.npy")
    except:
        ds = TNGDataset().unbatch().map(tf_func).take(484)
        array = list(ds.as_numpy_iterator())
        y_data = np.array([x[0] for x in array])
        x_data = np.array([x[1] for x in array])

        x_train = x_data[:train_size].reshape(-1, Nd[0], Nd[1])
        y_train = y_data[:train_size].reshape(-1, y_shape)
        x_test = x_data[train_size:len(x_data)-len(x_data)%batch_size].reshape(-1, Nd[0], Nd[1])
        y_test = y_data[train_size:len(x_data)-len(x_data)%batch_size].reshape(-1, y_shape)

        folder = f"./data/intermediate/TNG/{operator}/"
        os.makedirs(folder, exist_ok=True)

        np.save(f"{folder}/x_true_train_{ISNR}dB.npy",  x_train)
        np.save(f"{folder}/y_dirty_train_{ISNR}dB.npy", y_train)
        np.save(f"{folder}/x_true_test_{ISNR}dB.npy",   x_test)
        np.save(f"{folder}/y_dirty_test_{ISNR}dB.npy",  y_test)
elif data == "SATS":
    print("creating dataset")
    train_size = 300
    set_size = 300
    tf_func, func = measurement_func(uv,  m_op=None, Nd=(256,256), data_shape=y_shape, ISNR=ISNR)
    ds = SATSDataset()
    data_map = partial(data_map, y_size=y_shape, z_size=(256,256))
    # yogadl_dataset = make_yogadl_dataset(ds) # use yogadl for caching and shuffling the data
    dataset = ds.repeat(epochs).shuffle(set_size).map(center_crop).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)
    
    try:
        x_train = np.load(f"./data/intermediate/{data}/{operator}/x_true_train_{ISNR}dB.npy")
        y_train= np.load(f"./data/intermediate/{data}/{operator}/y_dirty_train_{ISNR}dB.npy")

        x_test = np.load(f"./data/intermediate/{data}/{operator}/x_true_test_{ISNR}dB.npy")
        y_test = np.load(f"./data/intermediate/{data}/{operator}/y_dirty_test_{ISNR}dB.npy")
    except:
        ds = ds.map(center_crop).map(tf_func).take(480)
        array = list(ds.as_numpy_iterator())
        y_data = np.array([x[0] for x in array])
        x_data = np.array([x[1] for x in array])

        x_train = x_data[:train_size].reshape(-1, Nd[0], Nd[1])
        y_train = y_data[:train_size].reshape(-1, y_shape)
        x_test = x_data[train_size:len(x_data)-len(x_data)%batch_size].reshape(-1, Nd[0], Nd[1])
        y_test = y_data[train_size:len(x_data)-len(x_data)%batch_size].reshape(-1, y_shape)

        folder = f"./data/intermediate/{data}/{operator}/"
        os.makedirs(folder, exist_ok=True)

        np.save(f"{folder}/x_true_train_{ISNR}dB.npy",  x_train)
        np.save(f"{folder}/y_dirty_train_{ISNR}dB.npy", y_train)
        np.save(f"{folder}/x_true_test_{ISNR}dB.npy",   x_test)
        np.save(f"{folder}/y_dirty_test_{ISNR}dB.npy",  y_test)


# create model
model = net(
    Nd, 
    uv=uv,
    op=op, 
    depth=4, 
    conv_layers=2,
    input_type='measurements', 
    measurement_weights=w,
    batch_size=batch_size
    )

def create_callbacks(data, operator, network, ISNR, postfix):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./models/{data}/{operator}/{network}_{ISNR}dB{postfix}" + "/cp-{epoch:04d}.ckpt",
        verbose=1, 
        save_weights_only=True,
        save_freq=save_freq* (set_size//batch_size)
    )

    csv_logger = CSV_logger_plus(f"./logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix)

    return [cp_callback, csv_logger]\

###### simple training ########
# train model using limited data
# print("start first model")
# postfix = f"_{data}"
# callbacks = create_callbacks(data, operator, network, ISNR, postfix)

# history = model.fit(
#     dataset,
#     epochs=epochs, 
#     callbacks=callbacks,
#     steps_per_epoch=set_size//batch_size
# )

# # predict using this model
# predict(x_train, y_train, x_test, y_test, model, batch_size, data, ISNR, postfix=postfix)


###### transfer predict w/o training ########
print("loading in coco model")
# load model 
postfix = f"_COCO"    
latest = tf.train.latest_checkpoint(f"./models/COCO/{operator}/{network}_{ISNR}dB")
model.load_weights(latest)
predict(x_train, y_train, x_test, y_test, model, batch_size, data, ISNR, postfix=postfix)

###### transfer learning ########
# retrain model
postfix = "_transfer"
callbacks = create_callbacks(data, operator, network, ISNR, postfix)

history = model.fit(
    dataset,
    epochs=epochs, 
    callbacks=callbacks,
    steps_per_epoch=set_size//batch_size
)

predict(x_train, y_train, x_test, y_test, model, batch_size, data, ISNR, postfix=postfix)



