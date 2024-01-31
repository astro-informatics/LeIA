from src.util import gpu_setup
gpu_setup()

import os
import sys
import time
import yaml

import numpy as np
import tensorflow as tf

from src.networks.UNet_var import UNet_var
from src.networks.GUNet_var import GUNet_var

from src.operators.NUFFT2D_TF import NUFFT2D_TF

from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 
from src.util import PSNRMetric
from src.data.RAI_datasets import Known, Fixed, Validation, Varying
from src.sampling.uv_sampling import spider_sampling, random_sampling
from src.dataset import PregeneratedDataset, data_map

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

dataset = PregeneratedDataset(
    operator=operator, epochs=epochs
    ).unbatch().batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: data_map(x,y, y_size=len(uv)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(train_size//batch_size)


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

### Train    
history = model.fit(
    dataset, 
    epochs=epochs, 
    callbacks=[csv_logger, cp_callback],
    steps_per_epoch=train_size//batch_size
)

pt_callback = PredictionTimeCallback(f"{results_base_folder}{dataset}/{operator}/summary_{network}{postfix}.csv", batch_size) 

### Predict
print("loading train and test data")

y_shape = len(uv)
x_true = np.load(data_folder+ f"x_true_train_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty = np.load(data_folder+ f"y_dirty_train_{ISNR}dB.npy").reshape(-1,y_shape)

x_true_test = np.load(data_folder+ f"x_true_test_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty_test = np.load(data_folder+ f"y_dirty_test_{ISNR}dB.npy").reshape(-1,y_shape)


print("predict train")
train_predict = model.predict(y_dirty, batch_size=batch_size, callbacks=[pt_callback])
print("predict test")
test_predict = model.predict(y_dirty_test, batch_size=batch_size)

print("saving train and test predictions")
os.makedirs(f"{data_base_folder}/processed/{dataset}/{operator}", exist_ok=True)
np.save(f"{data_base_folder}/processed/{dataset}/{operator}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
np.save(f"{data_base_folder}/processed/{dataset}/{operator}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)

### statistics
from src.util import calculate_statistics

statistics = calculate_statistics(x_true, train_predict, x_true_test, test_predict, operator, network, ISNR, postfix)
statistics.to_csv(f"{results_base_folder}/{dataset}/{operator}/statistics_{network}_{ISNR}dB{postfix}.csv")
