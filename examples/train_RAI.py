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


# if mode == "True":
#     uv = np.load(data_folder + "/uv_original.npy")
# else:
uv = np.load(data_folder + "/uv_big.npy")

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


### Datasets
yzx_known = Known(dataset, operator, seed=329248, ISNR=ISNR, batch_size=batch_size).unbatch().batch(batch_size)
yzx_fixed = Fixed(dataset, operator, seed=4620389, ISNR=ISNR, batch_size=batch_size).unbatch().batch(batch_size)
yzx_validation = Validation(dataset, operator, seed=562093, ISNR=ISNR, batch_size=batch_size).unbatch().batch(batch_size)
yzx_varying = Varying(dataset, operator, seed=65947, ISNR=ISNR, batch_size=batch_size).unbatch().batch(batch_size)

# transfer learn dataset
yzx_known_transfer = Known(dataset, operator, seed=329248, ISNR=ISNR, batch_size=batch_size, epochs=1).unbatch().batch(batch_size)

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
if mode == "True":
    print("Fitting to True visibility coverage")
    model.fit(yzx_known, steps_per_epoch=train_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)


elif mode == "Distribution" or mode == "Distribution transfer":
    model.fit(yzx_varying, steps_per_epoch=train_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)

    # transfer learning
    if mode == "Distribution transfer":
        csv_logger = CSV_logger_plus(f"{log_base_folder}/{dataset}/{operator}/log_{network}_{ISNR}dB" + postfix + "", append=True)

        print("fitting to known test sampling distribution")
        model.fit(yzx_known_transfer, steps_per_epoch=train_size//batch_size, epochs=transfer_epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)

elif mode == "Single" or mode == "Single transfer":
    print("fitting to known random sampling distribution")
    model.fit(yzx_fixed, steps_per_epoch=train_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)

    # transfer learning:
    if mode == "Single transfer":
        csv_logger = CSV_logger_plus(f"{log_base_folder}/{dataset}/{operator}/log_{network}_{ISNR}dB" + postfix + "", append=True)

        print("fitting to known test sampling distribution")
        model.fit(yzx_known_transfer, steps_per_epoch=train_size//batch_size, epochs=transfer_epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)

### Predict
y_shape = len(uv)

print("loading train and test data")
x_true = np.load(data_folder+ f"x_true_train_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty = np.load(data_folder+ f"y_dirty_train_{ISNR}dB.npy").reshape(-1,y_shape)
z = np.load(f"./data/intermediate/{dataset}/{operator}/sel.npy")            
sel_dirty = np.tile(z, len(y_dirty)).reshape(len(y_dirty), -1)

x_true_test = np.load(data_folder+ f"x_true_test_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty_test = np.load(data_folder+ f"y_dirty_test_{ISNR}dB.npy").reshape(-1,y_shape)
sel_dirty_test = np.tile(z, len(y_dirty_test)).reshape(len(y_dirty_test), -1)

latest = tf.train.latest_checkpoint(checkpoint_folder)
model.load_weights(latest)

print("predict train")
train_predict = model.predict([y_dirty, sel_dirty], batch_size=batch_size, callbacks=[])
print("predict test")
test_predict = model.predict([y_dirty_test, sel_dirty_test], batch_size=batch_size)

operator = "NUFFT_Random_var"

print("saving train and test predictions")
os.makedirs(f"{data_base_folder}/processed/{dataset}/{operator}", exist_ok=True)
np.save(f"{data_base_folder}/processed/{dataset}/{operator}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
np.save(f"{data_base_folder}/processed/{dataset}/{operator}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)

### statistics
from src.util import calculate_statistics

statistics = calculate_statistics(x_true, train_predict, x_true_test, test_predict, operator, network, ISNR, postfix)
statistics.to_csv(f"{results_base_folder}/{dataset}/{operator}/statistics_{network}_{ISNR}dB{postfix}.csv")

