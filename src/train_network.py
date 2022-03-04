
from functools import partial
from tensorflow.keras import callbacks
from src.network import Unet
# from src.networks.highlow2 import HighLowPassNet
from src.networks.highlow_fft import HighLowPassNet_fft as HighLowPassNet
from src.operators.discrete_fft_op import fft_op as NUFFT_op_TF

import numpy as np
import tensorflow as tf
import pickle 
import os
import time
import sys 

from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus
from src.dataset import * # TODO change this dirty import
from src.sampling.uv_sampling import spider_sampling

from util import gpu_setup
gpu_setup()

st = time.time()

ISNR = int(sys.argv[1])
network = sys.argv[2] # adjoint, unet, dunet
activation = sys.argv[3]
load_weights = bool(int(sys.argv[4]))
learned_adjoint = bool(int(sys.argv[5]))
learned_grad = bool(int(sys.argv[6]))
grad_on_upsample = bool(int(sys.argv[7]))
data = sys.argv[8]

# 30 adjoint sigmoid 0 0 0 0
# 30 adjoint sigmoid 0 1 0 0

# 30 unet sigmoid 0 0 0 0
# 30 unet sigmoid 0 1 0 0

# 30 dunet sigmoid 0 0 0 0
# 30 dunet sigmoid 0 1 0 0
# 30 dunet sigmoid 0 0 1 0
# 30 dunet sigmoid 0 0 0 1

# data = "COCO"
# data = "GZOO"
# data = "LLPS"
# data = "SATS"

set_size = 2000

train_time = 10*60 # time after which training should stop in mins
# i = "_same2"
i = "_unitary_discrete_sample"
# grad = False # 40x slower (27x)

postfix = "_" + activation

if learned_adjoint:
    postfix += "_learned_adjoint"
if learned_grad:
    postfix += "_learned_grad"
if grad_on_upsample:
    postfix += "_upsample_grad"

if i:
    postfix += str(i)

project_folder = os.environ["HOME"] + "/src_aiai/"
data_folder = project_folder + f"data/intermediate/{data}"
checkpoint_folder = project_folder+ f"models/{data}/{network}_{ISNR}dB{postfix}"
checkpoint_path = checkpoint_folder + "/cp-{epoch:04d}.ckpt"
tensorboard_logs_path = project_folder + "logs/tensorboard/"
log_folder = project_folder + "logs/"

def load_data(data_folder, ISNR=30):
    """load the pre-computed train and test data"""
    fft = lambda x: np.fft.fftshift(np.fft.fft2(
        np.fft.fftshift(x, axes=(1,2)), axes=(1,2), norm='ortho'), axes=(1,2))

    print("Loading train data")
    x_true = np.load(data_folder+ f"/x_true_train_{ISNR}dB.npy").reshape(-1,256,256)
    x_dirty = np.load(data_folder+ f"/x_dirty_train_{ISNR}dB.npy")
    y_dirty = np.load(project_folder + f"./data/intermediate/{data}/y_dirty_train_{ISNR}dB.npy").reshape( -1,4440)
    
    # print("Creating fft grid train")
    # y_dirty = fft(x_dirty)

    print("Loading test data")
    x_true_test = np.load( data_folder+ f"/x_true_test_{ISNR}dB.npy").reshape(-1,256,256)
    x_dirty_test = np.load( data_folder+ f"/x_dirty_test_{ISNR}dB.npy")
    y_dirty_test = np.load(project_folder + f"./data/intermediate/{data}/y_dirty_test_{ISNR}dB.npy").reshape( -1,4440)
    
    # print("Creating fft grid test")
    # y_dirty_test = fft(x_dirty_test)

    return x_true, x_dirty, y_dirty, x_true_test, x_dirty_test, y_dirty_test




def preprocess(x, m=None, s=None):
    """standardising data using a mean and standard deviation"""
    if not m:
        m, s = np.mean(x), np.std(x)
    x = (x - m)/s
    return x, m, s

def unpreprocess(x, m, s):
    return (x*s) + m

# preprocessing
# x_true_dirty, m, s = preprocess(x_true_dirty)
# x_test_dirty, *_ = preprocess(x_test_dirty)

epochs = 200
save_freq = 5
batch_size = 20

n_spokes = 37 
# n_spokes = 85 # Changed the ammount of spokes, is different from the prebuilt datasets
# uv = spider_sampling(n_spokes=n_spokes)
uv = (np.random.randn(2, 16384)).T
uv /= np.max(np.abs(uv)) * np.pi
input_size = (256,256,1)
# input_size = (128,128,1)


if network == "adjoint":
    depth = 0
    train_time = 4*60
    grad = False
elif network == "unet":
    depth = 4
    # depth = 2
    grad = False
elif network == "dunet":
    depth = 4
    # depth = 2
    grad = True
elif network == "highlow":
    depth=5
else:
    print("not valid network option")
    exit()


if network == "highlow":
    model = HighLowPassNet(
        input_size,
        uv=uv, 
        depth=depth, 
        start_filters=16, 
        conv_layers=3, 
        kernel_size=3, 
        conv_activation='relu', 
        output_activation=activation, 
    )
else:
    model = Unet(
        input_size,
        uv=uv, 
        depth=depth, 
        start_filters=16, 
        kernel_size=3, 
        conv_activation='relu', 
        output_activation=activation, 
        grad=grad,  
        learned_adjoint=learned_adjoint, 
        learned_grad=learned_grad, 
        grad_on_upsample=grad_on_upsample
    )





print("loading weights")
if load_weights:    
    latest = tf.train.latest_checkpoint(checkpoint_folder)
    model.load_weights(latest)
    csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/log_{network}_{ISNR}dB" + postfix + "", append=True)
else:
    csv_logger = CSV_logger_plus(log_folder +f"{data}/log_{network}_{ISNR}dB" + postfix + "")
    




cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=save_freq* (set_size//batch_size)
)


tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_folder, histogram_freq=10, write_graph=True,
    write_images=False, update_freq='epoch',
    profile_batch=2, embeddings_freq=0, embeddings_metadata=None
)


to_callback = TimeOutCallback(timeout=train_time, checkpoint_path=checkpoint_path)

callbacks = [
    cp_callback, 
    csv_logger, 
    # to_callback,
]

# early_stopping = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

m_op = NUFFT_op_TF()
m_op.plan(uv, Nd=input_size[:2],  Kd=(512, 512), Jd=(6,6), batch_size=20)

print("creating dataset")
tf_func, func = measurement_func(uv,  m_op=None, Nd=input_size[:2], ISNR=ISNR)
ds = Dataset(set_size, data)
data_map = partial(data_map, y_size=len(uv), z_size=input_size[:2])
yogadl_dataset = make_yogadl_dataset(ds) # use yogadl for caching and shuffling the data
if data in ["COCO", "SATS"]:
    dataset = ds.take(200).map(random_crop).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)
    # dataset = ds.take(200).map(random_crop).map(tf_func).batch(batch_size).map(data_map).snapshot("/mnt/mars/data/").prefetch(20)
elif data == "GZOO":
    dataset = ds.take(200).map(center_crop).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)
elif data == "LLPS":
    ds = EllipseDataset(set_size)
    yogadl_dataset = make_yogadl_dataset(ds)
    dataset = ds.take(200).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)

# data = ds.cache().map(crop).map(tf_func).map(set_shape).prefetch(tf.data.experimental.AUTOTUNE)

#TODO does yogadl shuffle in the 200 or just takes 200 out of 2000 shuffled items (should be latter)

print("training")
if network != 'adjoint' or learned_adjoint:
    history = model.fit(
        dataset, 
        epochs=epochs, 
        callbacks=callbacks
)

# history = model.fit(x=[x_dirty, y_dirty], 
#                     y=x_true, 
#                     epochs=epochs,
#                     batch_size=20, 
#                     # validation_data=(x_dirty_test, x_true_test),
#                     callbacks=[cp_callback, csv_logger, early_stopping])


pt_callback = PredictionTimeCallback(project_folder + f"/results/{data}/summary_{network}{postfix}.csv", batch_size) 

try:
    x_true, x_dirty, y_dirty, x_true_test, x_dirty_test, y_dirty_test = load_data(data_folder)
except: 
    pass

pickle.dump(history.history, open(project_folder + f"results/{data}/history_{network}_{ISNR}dB" + postfix + ".pkl", "wb"))

print("predict train")
train_predict = model.predict(y_dirty, batch_size=batch_size, callbacks=[pt_callback])
print("predict test")
test_predict = model.predict(y_dirty_test, batch_size=batch_size)

print("saving")
np.save(project_folder + f"data/processed/{data}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
np.save(project_folder + f"data/processed/{data}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)


y_dirty_test_robust = np.load(project_folder + "/data/intermediate/y_dirty_test_robustness.npy")
print("predict")
test_predict_robust = model.predict(y_dirty_test_robust, batch_size=20)
np.save(project_folder + f"data/processed/{data}/test_predict_{network}_robustness" + postfix + ".npy", test_predict_robust)

y_dirty_gen = np.load(project_folder + "/data/intermediate/y_dirty_gen_30dB.npy")
print("predict")
test_predict_gen = model.predict(y_dirty_gen, batch_size=20)
np.save(project_folder + f"data/processed/{data}/test_predict_{network}_gen" + postfix + ".npy", test_predict_gen)
# print("it took:",  (time.time()-st)/60, "mins")
