from functools import partial
from tensorflow.python.framework.ops import disable_eager_execution

from src.operators.NUFFT2D import NUFFT2D
disable_eager_execution()

import os
import sys 
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

# operators and sampling patterns
from src.operators.NUFFT2D_TF import NUFFT2D_TF

from src.sampling.uv_sampling import spider_sampling

 # some custom callbacks
from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 

# model and dataset generator
from src.networks.UNet import UNet
from src.dataset import Dataset, PregeneratedDataset, data_map, make_yogadl_dataset, measurement_func, random_crop

# selecting one gpu to train on
from src.util import gpu_setup
gpu_setup()


#TODO add a nice argument parser

epochs = 200
set_size = 2000 # size of the train set
save_freq = 20 # save every 20 epochs
batch_size = 20 


ISNR = 30 #dB
network = "UNet"
activation = "linear"
load_weights = bool(int(sys.argv[1])) # continuing the last run
operator = "NUFFT_SPIDER"
data = "COCO"

try: 
    postfix = "_" + str(sys.argv[3])
except:
    postfix = ""


im_shape = (256, 256)
upsampling = 2
op = NUFFT2D_TF
uv = spider_sampling()
y_shape = len(uv)

def calculate_measurement_weighting(uv):
    # sampling density based weighting
    grid_cell = 2*np.pi / im_shape[0]*upsampling
    binned = (uv[:,:]+np.pi+.5*grid_cell) // grid_cell
    binned = [tuple(x) for x in binned]
    cells = set(binned)
    w_gridded = np.zeros(uv.shape[0])
    for cell in list(cells):
        mask = np.all(np.array(cell) ==  binned, axis=1)
        w_gridded[mask] = np.sum(mask)

    w = 1/w_gridded
    w /= w.max()
    return w

measurement_weights = calculate_measurement_weighting(uv)


if network == "UNet":
    net = UNet()
else:
    print("select a valid network to train")
    exit()
    
model = net(
    im_shape, 
    uv=uv,
    op=op, 
    depth=4, 
    conv_layers=2,
    input_type="measurements", 
    measurement_weights=measurement_weights,
    batch_size=batch_size
    )


def create_train_dataset(data, operator, epochs, batch_size, set_size, y_shape):
    """Tries to use pre-augmented data, otherwise creates a new dataset with augmentation"""
    try:  
        dataset = PregeneratedDataset(
        operator=operator, epochs=epochs
        ).unbatch().batch(
            batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            lambda x, y: data_map(x,y, y_size=len(uv)), 
            num_parallel_calls=tf.data.AUTOTUNE
            ).prefetch(set_size//batch_size)
    except:
        tf_func, _ = measurement_func(uv,  m_op=op, Nd=(256,256), data_shape=y_shape, ISNR=ISNR)
        ds = Dataset(set_size, data)
        yogadl_dataset = make_yogadl_dataset(ds) # use yogadl for caching and shuffling the data
        data_map = partial(data_map, y_size=y_shape, z_size=(256,256))
        dataset = yogadl_dataset.map(random_crop).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)
   
    return dataset

dataset = create_train_dataset(data, operator, epochs, batch_size, set_size, y_shape)

# defining the necessary paths based on parameters
project_folder = os.environ["HOME"] + "/src_aiai/"
data_folder = project_folder + f"data/intermediate/{data}/{operator}/"
checkpoint_folder = project_folder+ f"models/{data}/{operator}/{network}_{ISNR}dB{postfix}"
checkpoint_path = checkpoint_folder + "/cp-{epoch:04d}.ckpt"


if load_weights:    
    print("loading weights")
    latest = tf.train.latest_checkpoint(checkpoint_folder)
    model.load_weights(latest)
    csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix + "", append=True)
else:
    csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix + "")


def create_callbacks(): 
    # for saving the model
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=save_freq* (set_size//batch_size)
    )

    # for making sure the model is saved before hpc job times out
    to_callback = TimeOutCallback(timeout=max_train_time, checkpoint_path=checkpoint_path)
    tbc = tf.keras.callbacks.TensorBoard(log_dir='./tmp/tb')

    callbacks = [
        cp_callback, 
        csv_logger, 
        tbc
        # to_callback,
    ]
    return callbacks

callbacks = create_callbacks()


print("training")

if not load_weights:
    history = model.fit(
        dataset,
        epochs=epochs, 
        callbacks=callbacks,
        steps_per_epoch=set_size//batch_size
    )

def predict():

    # for saving how long predictions take
    pt_callback = PredictionTimeCallback(project_folder + f"/results/{data}/{operator}/summary_{network}{postfix}.csv", batch_size) 


    print("Saving model history")
    pickle.dump(history.history, open(project_folder + f"results/{data}/history_{network}_{ISNR}dB" + postfix + ".pkl", "wb"))
    #TODO add robustness test to this
    y_dirty_robustness = np.load(data_folder+ f"y_dirty_test_{ISNR}dB_robustness.npy").reshape(-1,y_shape)
    robustness_predict = model.predict(y_dirty_robustness, batch_size=batch_size, callbacks=[pt_callback])
    np.save(project_folder + f"data/processed/{data}/{operator}/test_predict_{network}_{ISNR}dB" + postfix + "_robustness.npy", robustness_predict)


    print("loading train and test data")
    x_true = np.load(data_folder+ f"x_true_train_{ISNR}dB.npy").reshape(-1,256,256)
    y_dirty = np.load(data_folder+ f"y_dirty_train_{ISNR}dB.npy").reshape(-1,y_shape)

    x_true_test = np.load(data_folder+ f"x_true_test_{ISNR}dB.npy").reshape(-1,256,256)
    y_dirty_test = np.load(data_folder+ f"y_dirty_test_{ISNR}dB.npy").reshape(-1,y_shape)


    print("predict train")
    train_predict = model.predict(y_dirty, batch_size=batch_size, callbacks=[pt_callback])
    print("predict test")
    test_predict = model.predict(y_dirty_test, batch_size=batch_size)

    print("saving train and test predictions")
    os.makedirs(project_folder + f"data/processed/{data}/{operator}", exist_ok=True)
    np.save(project_folder + f"data/processed/{data}/{operator}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
    np.save(project_folder + f"data/processed/{data}/{operator}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)

    return x_true, train_predict, x_true_test, test_predict

x_true, train_predict, x_true_test, test_predict = predict()


def calculate_statistics(x_true, train_predict, x_true_test, test_predict):
    import pandas as pd
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

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

    statistics.to_csv(project_folder + f"results/{data}/{operator}/statistics_{network}_{ISNR}dB{postfix}.csv")

calculate_statistics(x_true, train_predict, x_true_test, test_predict)