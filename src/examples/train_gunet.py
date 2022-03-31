
import os
import sys 
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

tf.compat.v1.disable_eager_execution() # GUNet cannot use eager execution

# operators and sampling patterns
from src.operators.NUFFT2D_TF import NUFFT2D_TF
from src.operators.NNFFT2D_TF import NNFFT2D_TF
from src.sampling.uv_sampling import spider_sampling, random_sampling

 # some custom callbacks
from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 

# model and dataset generator
from src.networks.GUnet import GUnet
from src.dataset import PregeneratedDataset, data_map

# selecting one gpu to train on
from src.util import gpu_setup
gpu_setup()


#TODO add a nice argument parser

epochs = 200
set_size = 2000 # size of the train set
save_freq = 20 # save every 20 epochs
batch_size = 20 
max_train_time = 40*60 # time after which training should stop in mins


ISNR = 30 #dB
network = "GUnet"
activation = "linear"
load_weights = bool(int(sys.argv[1])) # continuing the last run
operator = str(sys.argv[2])

try: 
    postfix = "_" + str(sys.argv[3])
except:
    postfix = ""


data = "COCO"

Nd = (256, 256)
Kd = (512, 512)
Jd = (6,6)

if operator == "NUFFT_SPIDER":
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
elif operator == "NUFFT_Random":
    y_shape = int(Nd[0]**2/2)
    uv = random_sampling(y_shape)
    op = NUFFT2D_TF
    w = np.ones(len(uv)) # no weights necessary for 50% sampling
elif operator == "NNFFT_Random":
    y_shape = int(Nd[0]**2/2)
    uv = random_sampling(y_shape)
    op = NNFFT2D_TF
    w = np.ones(len(uv)) # no weights necessary for 50% sampling
else:
    print("No such operator")
    exit()



model = GUnet(
    Nd, 
    uv=uv,
    op=op, 
    depth=4, 
    input_type="measurements", 
    measurement_weights=w,
    batch_size=batch_size
    )

if not load_weights: 
    dataset = PregeneratedDataset(
    operator=operator, epochs=epochs
    ).unbatch().batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE).map(lambda x, y: data_map(x,y, y_size=len(uv)), num_parallel_calls=tf.data.AUTOTUNE).prefetch(set_size//batch_size)



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
    

# for saving the model
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=save_freq* (set_size//batch_size)
)

# for making sure the model is saved before hpc job times out
to_callback = TimeOutCallback(timeout=max_train_time, checkpoint_path=checkpoint_path)

callbacks = [
    cp_callback, 
    csv_logger, 
    # to_callback,
]


print("training")

if not load_weights:
    history = model.fit(
        dataset, 
        epochs=epochs, 
        callbacks=callbacks,
        steps_per_epoch=set_size//batch_size
    )


# for saving how long predictions take
pt_callback = PredictionTimeCallback(project_folder + f"/results/{data}/{operator}/summary_{network}{postfix}.csv", batch_size) 

#TODO add robustness test to this
y_dirty_robustness = np.load(data_folder+ f"y_dirty_test_{ISNR}dB_robustness.npy").reshape(-1,y_shape)
robustness_predict = model.predict(y_dirty_robustness, batch_size=batch_size, callbacks=[pt_callback])
np.save(project_folder + f"data/processed/{data}/{operator}/test_predict_{network}_{ISNR}dB" + postfix + "_robustness.npy", robustness_predict)

print("Saving model history")
pickle.dump(history.history, open(project_folder + f"results/{data}/history_{network}_{ISNR}dB" + postfix + ".pkl", "wb"))

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

#TODO add robustness test to this

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

statistics.to_csv(project_folder + f"results/{data}/{operator}/statistics_{network}{postfix}.csv")
