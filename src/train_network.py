
from src.network import Unet
import numpy as np
import tensorflow as tf
import pickle 
import os
import time
import sys 

from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus
from src.dataset import *
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
data = "LLPS"

set_size = 2000

train_time = 10*60 # time after which training should stop in mins
i = 0
# grad = False # 40x slower (27x)

postfix = "_" + activation

if learned_adjoint:
    postfix += "_learned_adjoint"
if learned_grad:
    postfix += "_learned_grad"
if grad_on_upsample:
    postfix += "_upsample_grad"

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
    y_dirty_test = np.load(project_folder + f"./data/intermediate/{data}/y_dirty_train_{ISNR}dB.npy").reshape( -1,4440)
    
    # print("Creating fft grid test")
    # y_dirty_test = fft(x_dirty_test)

    return x_true, x_dirty, y_dirty, x_true_test, x_dirty_test, y_dirty_test

try:
    x_true, x_dirty, y_dirty, x_true_test, x_dirty_test, y_dirty_test = load_data(data_folder)
except: 
    pass


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
# set_size = 200 # TODO

uv = spider_sampling()

if network == "adjoint":
    depth = 0
    train_time = 4*60
    grad = False
elif network == "unet":
    depth = 4
    grad = False
elif network == "dunet":
    depth = 4
    grad = True
else:
    print("not valid network option")
    exit()

model = Unet(
    (256,256,1), 
    uv=uv, 
    depth=depth, 
    start_filters=16, 
    conv_layers=3, 
    kernel_size=3, 
    conv_activation='relu', 
    output_activation=activation, 
    grad=grad, 
    learned_adjoint=learned_adjoint, 
    learned_grad=learned_grad, 
    grad_on_upsample=grad_on_upsample
)



# print(model.summary())
# exit()


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

# early_stopping = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

print("creating dataset")
tf_func, func = measurement_func(ISNR=ISNR)
ds = Dataset(set_size)
yogadl_dataset = make_yogadl_dataset(ds) # use yogadl for caching and shuffling the data
if data == "COCO":
    dataset = ds.take(200).map(random_crop).map(tf_func).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)
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
        callbacks=[cp_callback, csv_logger, to_callback]
)

# history = model.fit(x=[x_dirty, y_dirty], 
#                     y=x_true, 
#                     epochs=epochs,
#                     batch_size=20, 
#                     # validation_data=(x_dirty_test, x_true_test),
#                     callbacks=[cp_callback, csv_logger, early_stopping])


pt_callback = PredictionTimeCallback(project_folder + f"/results/{data}/summary.csv", batch_size) 

print("predict train")
train_predict = model.predict(y_dirty, batch_size=batch_size, callbacks=[pt_callback])
print("predict test")
test_predict = model.predict(y_dirty_test, batch_size=batch_size)

print("saving")
np.save(project_folder + f"data/processed/{data}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
np.save(project_folder + f"data/processed/{data}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)
pickle.dump(history.history, open(project_folder + f"results/{data}/history_{network}_{ISNR}dB" + postfix + ".pkl", "wb"))


# print("it took:",  (time.time()-st)/60, "mins")
