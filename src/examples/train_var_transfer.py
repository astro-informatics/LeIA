from functools import partial

import os
import sys 
import pickle
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks


# operators and sampling patterns
from src.operators.NUFFT2D import NUFFT2D
from src.operators.NUFFT2D_TF import NUFFT2D_TF
from src.sampling.uv_sampling import spider_sampling, random_sampling

 # some custom callbacks
from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 

# model and dataset generator
from src.networks.UNet_var import UNet_var
from src.networks.GUNet_var import GUNet_var
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


operator = "NUFFT_Random_var"
data = "TNG"
ISNR = 30  
postfix = ""
network = "UNet_var"
# network = "GUNet_var"

epochs= 100
set_size = 2000
train_size = 300
test_size = 180
save_freq = 20

Nd = (256,256)
# batch_size = 20
batch_size = 5

project_folder = os.environ["HOME"] + "/src_aiai/"
data_folder = project_folder + f"data/intermediate/{data}/{operator}/"

uv = np.load("./data/intermediate/COCO/NUFFT_Random_var/uv_big.npy")
uv_test = np.load(data_folder + "/uv_original.npy")

mode = int(sys.argv[1])
if mode == 0:
    # known -> known
    uv = uv_test
    postfix = "_known"
    coco_checkpoint_folder = project_folder+ f"models/COCO/{operator}/{network}_{ISNR}dB_known"
elif mode == 1:
    # generalized -> generalized
    postfix = "_generalized"
    coco_checkpoint_folder = project_folder+ f"models/COCO/{operator}/{network}_{ISNR}dB_generalized"

model = UNet_var(
    Nd, 
    uv=uv,
    op=NUFFT2D_TF, 
    depth=4, 
    conv_layers=2,
    input_type="measurements", 
    measurement_weights=np.ones(len(uv)),
    batch_size=batch_size,
    residual=True
    )

# model = GUNet_var(
#     Nd, 
#     uv=uv,
#     op=NUFFT2D_TF, 
#     depth=4, 
#     conv_layers=2,
#     input_type="measurements", 
#     measurement_weights=np.ones(len(uv)),
#     batch_size=batch_size,
#     residual=True
#     )

yshape  = len(uv)

checkpoint_folder = project_folder+ f"models/{data}/{operator}/{network}_{ISNR}dB{postfix}"
checkpoint_path = checkpoint_folder + "/cp-{epoch:04d}.ckpt"

print("starting model building and compilation")
st = time.time()


# create dataset
class TNGDataset(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator():
        files = glob.glob("/home/mars/git/IllustrisTNG/data/processed_256/TNG*.npy")
        x = np.array([np.load(file) for file in files])
        while True:
            yield x[:,:,:,np.newaxis]

    def __new__(cls):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float32),
            args=()
        )




train_size = 2000
test_size = 1000
set_size = train_size


# creating train and test set 
try:
    x_train = np.load(f"./data/intermediate/TNG/{operator}/x_true_train_{ISNR}dB.npy")
    y_train= np.load(f"./data/intermediate/TNG/{operator}/y_dirty_train_{ISNR}dB.npy")

    x_test = np.load(f"./data/intermediate/TNG/{operator}/x_true_test_{ISNR}dB.npy")
    y_test = np.load(f"./data/intermediate/TNG/{operator}/y_dirty_test_{ISNR}dB.npy")
except:
    # use only the first 300 for training
    tf_func_train, func = measurement_func(uv_test,  m_op=None, Nd=(256,256), data_shape=len(uv_test), ISNR=ISNR)
    tf_func_test, func = measurement_func(uv_test,  m_op=None, Nd=(256,256), data_shape=len(uv_test), ISNR=ISNR)

    ds_train = TNGDataset().unbatch().take(train_size).map(tf_func_train)
    array = list(ds_train.as_numpy_iterator())
    x_train = np.array([x[1] for x in array])
    y_train = np.array([x[0] for x in array])

    ds_test = TNGDataset().unbatch().skip(train_size).take(test_size).map(tf_func_test)
    array = list(ds_test.as_numpy_iterator())
    x_test = np.array([x[1] for x in array])
    y_test = np.array([x[0] for x in array])

    folder = f"./data/intermediate/TNG/{operator}/"
    os.makedirs(folder, exist_ok=True)

    np.save(f"{folder}/x_true_train_{ISNR}dB.npy",  x_train)
    np.save(f"{folder}/y_dirty_train_{ISNR}dB.npy", y_train)
    np.save(f"{folder}/x_true_test_{ISNR}dB.npy",   x_test)
    np.save(f"{folder}/y_dirty_test_{ISNR}dB.npy",  y_test)




def create_callbacks(data, operator, network, ISNR, postfix):
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"./models/{data}/{operator}/{network}_{ISNR}dB{postfix}" + "/cp-{epoch:04d}.ckpt",
        verbose=1, 
        save_weights_only=True,
        save_freq=save_freq* (set_size//batch_size)
    )

    csv_logger = CSV_logger_plus(f"./logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix)

    return [cp_callback, csv_logger]

callbacks = create_callbacks(data, operator, network, ISNR, postfix)

class X(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, operator="NUFFT_SPIDER", ISNR=30):
        i = 0
        try:
            operator = operator.decode('utf-8')
        except:
            pass
        while True:
            x = np.load(f"./data/intermediate/COCO/{operator}/x_true_train_{ISNR}dB_{i:03d}.npy")
          
            yield x
            i = (i + 1) % 100 # only a 100 presaved so reuse them

    def __new__(cls, operator, ISNR=30, epochs=100):
        # assert os.path.exists(
            # f"./data/intermediate/COCO/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float32),
            args=(epochs, operator, ISNR)
        )
    
class Y(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, operator="NUFFT_SPIDER", ISNR=30):
        i = 0
        try:
            operator = operator.decode('utf-8')
        except:
            pass
        while True:
            y = np.load(f"./data/intermediate/COCO/{operator}/y_dirty_train_{ISNR}dB_{i:03d}.npy")
            yield y
            i = (i + 1) % 100 # only a 100 presaved so reuse them

    def __new__(cls, operator, ISNR=30, epochs=100):
        # assert os.path.exists(
            # f"./data/intermediate/COCO/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.complex64),
            args=(epochs, operator, ISNR)
        )

class Z(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, operator="NUFFT_SPIDER", ISNR=30):
        while True:
            p = np.random.permutation(len(uv)) < len(uv)//2
            # z = np.random.randint(0, 2, size=len(uv), dtype=bool)
            yield np.repeat(p, batch_size).reshape(-1, batch_size).T
            
    def __new__(cls, operator, ISNR=30, epochs=100):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.bool),
            args=(epochs, operator, ISNR)
        )

class F(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, operator="NUFFT_SPIDER", ISNR=30):
        while True:
            z = np.ones((set_size, len(uv)), dtype=bool)
            yield z

    def __new__(cls, operator, ISNR=30, epochs=100):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.bool),
            args=(epochs, operator, ISNR)
        )
class X(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, operator="NUFFT_SPIDER", ISNR=30):
        i = 0
        try:
            operator = operator.decode('utf-8')
        except:
            pass
        while True:
            x = np.load(f"./data/intermediate/COCO/{operator}/x_true_train_{ISNR}dB_{i:03d}.npy")
            
            yield x
            i = (i + 1) % 100 # only a 100 presaved so reuse them

    def __new__(cls, operator, ISNR=30, epochs=100):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.float32),
            args=(epochs, operator, ISNR)
        )   

class SelDataset(tf.data.Dataset):
    @staticmethod
    def _generator(epochs):
        np.random.seed(427483)
        while True:
            p = np.random.permutation(len(uv)) < len(uv)//2
            # z = np.random.randint(0, 2, size=len(uv), dtype=bool)
            yield np.repeat(p, set_size).reshape(-1, set_size).T

    def __new__(cls, epochs=100):

        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.bool),
            args=(epochs,)
        )
# x_dataset = tf.data.Dataset.from_tensor_slices(x_train).repeat(epochs)
# y_dataset = tf.data.Dataset.from_tensor_slices(y_train).repeat(epochs)
 
# should maybe shuffle datasets
x_dataset = X(
    operator=operator, epochs=epochs
    ).unbatch()

y_dataset = Y(
    operator=operator, epochs=epochs
    ).unbatch()

z_dataset = Z(
    operator=operator, epochs=epochs
    ).unbatch()

f_dataset = F(
    operator=operator, epochs=epochs
    ).unbatch()

val_sel = SelDataset(
    epochs=epochs  
).unbatch()

if mode == 0:
    x_dataset = tf.data.Dataset.from_tensor_slices(x_train).repeat(epochs)
    y_dataset = tf.data.Dataset.from_tensor_slices(y_train).repeat(epochs)

    yz_known = tf.data.Dataset.zip((y_dataset, f_dataset))
    yzx_known = tf.data.Dataset.zip((yz_known, x_dataset)).batch(batch_size)

    latest = tf.train.latest_checkpoint(checkpoint_folder)
    model.load_weights(latest)

    # model.fit(yzx_known, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=callbacks)
elif mode == 1:
    yz_known = tf.data.Dataset.zip((y_dataset, f_dataset))
    yzx_known = tf.data.Dataset.zip((yz_known, x_dataset)).batch(batch_size)

    yz_varying = tf.data.Dataset.zip((y_dataset, z_dataset))
    yzx_varying = tf.data.Dataset.zip((yz_varying, x_dataset)).batch(batch_size)

    # validation dataset
    yz_validation = tf.data.Dataset.zip((y_dataset, val_sel.skip(set_size).take(set_size))) 
    yzx_validation = tf.data.Dataset.zip((yz_validation, x_dataset)).batch(batch_size)


    latest = tf.train.latest_checkpoint(checkpoint_folder)
    model.load_weights(latest)

    # model.fit(yzx_varying, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=callbacks)
    # model.fit(yzx_varying, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=callbacks, validation_data=yzx_validation, validation_steps=set_size//batch_size)



### evaluation

uv_test = np.load(data_folder + "/uv_original.npy")
y_shape = len(uv_test)

print("loading train and test data")
x_true = np.load(data_folder+ f"x_true_train_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty = np.load(data_folder+ f"y_dirty_train_{ISNR}dB.npy").reshape(-1,len(uv_test))
sel_dirty = np.ones_like(y_dirty, dtype=bool)
# sel_dirty = np.repeat( np.random.permutation(len(uv)) < len(uv)//2, len(y_dirty)).reshape(len(uv), len(y_dirty)).T.astype(bool)


x_true_test = np.load(data_folder+ f"x_true_test_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty_test = np.load(data_folder+ f"y_dirty_test_{ISNR}dB.npy").reshape(-1,len(uv_test))
sel_dirty_test = np.ones_like(y_dirty_test, dtype=bool)
# sel_dirty_test = np.repeat( np.random.permutation(len(uv)) < len(uv)//2, len(y_dirty_test)).reshape(len(uv), len(y_dirty_test)).T.astype(bool)

# m_op = NUFFT2D_TF()
# m_op.plan(uv_test, (Nd[0], Nd[1]), (Nd[0]*2, Nd[1]*2), (6,6))

model = model.rebuild_with_op(uv_test)

print("predict train")
train_predict = model.predict([y_dirty, sel_dirty], batch_size=batch_size, callbacks=[])

print("predict test")
test_predict = model.predict([y_dirty_test, sel_dirty_test], batch_size=batch_size)

operator = "NUFFT_Random_var"


print("saving train and test predictions")
os.makedirs(project_folder + f"data/processed/{data}/{operator}", exist_ok=True)
np.save(project_folder + f"data/processed/{data}/{operator}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
np.save(project_folder + f"data/processed/{data}/{operator}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)


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
