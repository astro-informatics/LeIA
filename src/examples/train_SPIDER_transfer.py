from src.util import gpu_setup
gpu_setup()

import os
import sys
import time
import yaml

import numpy as np
import tensorflow as tf

from functools import partial

from src.networks.UNet_var import UNet_var
from src.networks.GUNet_var import GUNet_var

from src.operators.NUFFT2D_TF import NUFFT2D_TF

from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 
from src.util import PSNRMetric, calculate_statistics
from src.data.RAI_datasets import Known, Fixed, Validation, Varying
from src.sampling.uv_sampling import spider_sampling, random_sampling
from src.dataset import PregeneratedDataset, data_map
from src.data.SPIDER_datasets import TNGDataset, SATSDataset
from src.dataset import Dataset, PregeneratedDataset, center_crop, data_map, make_yogadl_dataset, measurement_func, random_crop, data_map_image

config_file = str(sys.argv[1])
with open(config_file, "r") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

### load values from config file
operator = cfg.get("operator", "NUFFT_Random_var")
dataset = cfg.get("dataset", "TNG")
network = cfg.get("network", "UNet")
ISNR = cfg.get("ISNR", 30)

mode = cfg.get("training_strategy", "True")
exp_name = cfg.get("exp_name", "test")

data_base_folder = cfg.get("data_base_folder", "./data/")
checkpoint_base_folder = cfg.get("checkpoint_base_folder", "./models/")
log_base_folder = cfg.get("log_base_folder", "./logs/")
results_base_folder = cfg.get("results_base_folder", "./results/")

Nd = cfg.get("Nd", 256) # input image size
Kd = cfg.get("Kd", 512) # upsampled image size (for use in NUFFT operator)
Jd = cfg.get("Jd", 6)   # NUFFT kernel size

epochs = cfg.get("epochs", 100)
transfer_epochs = cfg.get("transfer_epochs", 100)

batch_size = cfg.get("batch_size", 2)
train_size = cfg.get("train_size", 2000)
val_size = cfg.get("val_size", 1000)
test_size = cfg.get("test_size", 1000)
save_freq = cfg.get("save_freq", 5)

postfix = f"_{mode}_{exp_name}"

data_folder = f"{data_base_folder}/intermediate/{dataset}/{operator}/"
checkpoint_folder = f"{checkpoint_base_folder}/{dataset}/{operator}/{network}_{ISNR}dB_{mode}_{exp_name}"
checkpoint_path = checkpoint_folder + "/cp-{epoch:04d}.ckpt"


# Calculate sampling and measurement weighting
uv = spider_sampling()
grid_cell = 2*np.pi / Kd 
binned = (uv[:,:]+np.pi+.5*grid_cell) // grid_cell
binned = [tuple(x) for x in binned]
cells = set(binned)
w_gridded = np.zeros(uv.shape[0])
for cell in list(cells):
    mask = np.all(np.array(cell) ==  binned, axis=1)
    w_gridded[mask] = np.sum(mask)

w = 1/w_gridded
w /= w.max()


### Setup
if network == "UNet":
    model = UNet_var(
    (Nd, Nd), 
    uv=uv,
    op=NUFFT2D_TF, 
    depth=4, 
    conv_layers=2,
    input_type="measurements", 
    measurement_weights=1,
    batch_size=batch_size,
    residual=False,
    metrics=[PSNRMetric()]
    )
elif network == "GUNet":
    model = GUNet_var(
    (Nd, Nd), 
    uv=uv,
    op=NUFFT2D_TF, 
    depth=4, 
    conv_layers=2,
    input_type="measurements", 
    measurement_weights=np.ones(len(uv)),
    batch_size=batch_size,
    residual=False,
    metrics=[PSNRMetric()]
    )

tf_func, func = measurement_func(uv,  m_op=None, Nd=(256,256), data_shape=len(uv), ISNR=ISNR)
if dataset == "TNG":
    ds = TNGDataset().unbatch().take(train_size).cache()
    # yogadl_dataset = make_yogadl_dataset(ds) # for efficient shuffling
    dataset = ds.repeat(epochs).shuffle(train_size).map(center_crop).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)
elif dataset == "SATS":
    ds = SATSDataset()
    data_map = partial(data_map, y_size=len(uv), z_size=(256,256))
    dataset = ds.repeat(epochs).shuffle(train_size).map(center_crop).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)
else:
    print(f"Invalid dataset: {dataset}")
    exit()

### Callbacks
csv_logger = CSV_logger_plus(f"{log_base_folder}/{dataset}/{operator}/log_{network}_{ISNR}dB" + postfix + "")


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    monitor='val_PSNR',
    mode='max',
    save_best_only='True',
    save_weights_only=True,
    save_freq='epoch'#save_freq* (train_size//batch_size)
)

def predict_and_statistics(x_true, y_dirty, x_true_test, y_dirty_test, model, batch_size, data, ISNR, postfix=""):
    print("predict train")
    train_predict = model.predict(y_dirty, batch_size=batch_size)
    print("predict test")
    test_predict = model.predict(y_dirty_test, batch_size=batch_size)

    print("saving train and test predictions")
    os.makedirs(f"./data/processed/{data}/{operator}", exist_ok=True)
    np.save(f"./data/processed/{data}/{operator}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
    np.save(f"./data/processed/{data}/{operator}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)

    calculate_statistics(x_true, train_predict, x_true_test, test_predict, operator, network, ISNR, postfix)


x_train = np.load(f"{data_folder}/x_true_train_{ISNR}dB.npy")
y_train= np.load(f"{data_folder}/y_dirty_train_{ISNR}dB.npy")

x_test = np.load(f"{data_folder}/x_true_test_{ISNR}dB.npy")
y_test = np.load(f"{data_folder}/y_dirty_test_{ISNR}dB.npy")


### Train and Predict on TNG/SATS data  
history = model.fit(
    dataset, 
    epochs=epochs, 
    callbacks=[csv_logger, cp_callback],
    steps_per_epoch=train_size//batch_size
)

postfix += f"_{dataset}"
predict_and_statistics(x_train, y_train, x_test, y_test, model, batch_size, dataset, ISNR, postfix)


### Predict using COCO trained model
postfix += "_COCO"    
latest = tf.train.latest_checkpoint(f"./models/COCO/{operator}/{network}_{ISNR}dB")
model.load_weights(latest)
predict_and_statistics(x_train, y_train, x_test, y_test, model, batch_size, dataset, ISNR, postfix)


### Transfer Learning and Predict using TNG/SATS data
postfix += "_transfer"

history = model.fit(
    dataset,
    epochs=epochs, 
    callbacks=[csv_logger, cp_callback],
    steps_per_epoch=train_size//batch_size
)

predict_and_statistics(x_train, y_train, x_test, y_test, model, batch_size, dataset, ISNR, postfix)
