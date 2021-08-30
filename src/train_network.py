
from src.network import *
import numpy as np
import tensorflow as tf
import pickle 
import os
import time
import sys 

from src.dataset import *
from src.sampling.uv_sampling import spider_sampling

st = time.time()

ISNR = int(sys.argv[1])
network = sys.argv[2]
data = sys.argv[3]
activation = sys.argv[4]
load_weights = bool(int(sys.argv[5]))
grad = bool(int(sys.argv[6]))
set_size = int(sys.argv[7])

train_time = 10*60 # time after which training should stop in mins
i = 4
# grad = False # 40x slower (27x)
if grad:
    postfix = "_" + activation + "_grad_new_" + str(i)
else:
    postfix = "_" + activation + "_new_" + str(i)


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
    x_true = np.load(data_folder+ f"/x_true_train_{ISNR}dB.npy")
    x_dirty = np.load(data_folder+ f"/x_dirty_train_{ISNR}dB.npy")
    y_dirty = np.load(project_folder + f"./data/intermediate/{data}/y_dirty_train_{ISNR}dB.npy").reshape( -1,4440)
    
    # print("Creating fft grid train")
    # y_dirty = fft(x_dirty)

    print("Loading test data")
    x_true_test = np.load( data_folder+ f"/x_true_test_{ISNR}dB.npy")
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


uv = spider_sampling()



if network == 'small':
    model = small_unet(
    output_activation = activation,
    grad = grad
    )
else:
    model = medium_unet([256,256,1], uv, 
    output_activation = activation,
    grad = grad
    )

# print(model.summary())
# exit()

print("loading weights")
if load_weights:    
    latest = tf.train.latest_checkpoint(checkpoint_folder)
    model.load_weights(latest)
    csv_logger = tf.keras.callbacks.CSVLogger(project_folder + f"logs/{data}/log_{network}_{ISNR}dB" + postfix + "", append=True)
else:
    csv_logger = tf.keras.callbacks.CSVLogger(log_folder +f"{data}/log_{network}_{ISNR}dB" + postfix + "")
    


epochs = 1000
save_freq = 5
batch_size = 20
# set_size = 200 # TODO

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

class TimeOut(tf.keras.callbacks.Callback):
    """Based on https://stackoverflow.com/questions/58096219/save-a-tensorflow-model-after-a-fixed-training-time"""
    def __init__(self, t0, timeout, checkpoint_path):
        super().__init__()
        self.t0 = t0
        self.timeout = timeout  # time in minutes
        self.checkpoint_path = checkpoint_path

    def on_train_epoch_end(self, epoch, logs=None):
        if time.time() - self.t0 > self.timeout * 60: 
            print(f"\nReached {(time.time() - self.t0) / 60:.3f} minutes of training, stopping")
            self.model.stop_training = True
            self.model.save_weights(self.checkpoint_path.format(epoch=epoch+1))


to_callback = TimeOut(t0=time.time(), timeout=train_time, checkpoint_path=checkpoint_path)

# early_stopping = tf.keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

print("creating dataset")
tf_func, func = measurement_func(ISNR=ISNR)
ds = Dataset(set_size)
dataset = ds.cache().map(crop).map(tf_func).shuffle(set_size, reshuffle_each_iteration=True).batch(batch_size).map(data_map).prefetch(tf.data.experimental.AUTOTUNE)
# data = ds.cache().map(crop).map(tf_func).map(set_shape).prefetch(tf.data.experimental.AUTOTUNE)

#TODO does the fitting just ignore additional inputs?

print("training")
history = model.fit(dataset, 
                    
#                     batch_size=32, 
                    epochs=epochs, 
#                     validation_data=((x_dirty_test, y_dirty_test), x_true_test),
#                     validation_steps=50,
#                     validation_batch_size=20,
                    callbacks=[cp_callback, csv_logger, to_callback]
)

# history = model.fit(x=[x_dirty, y_dirty], 
#                     y=x_true, 
#                     epochs=epochs,
#                     batch_size=20, 
#                     # validation_data=(x_dirty_test, x_true_test),
#                     callbacks=[cp_callback, csv_logger, early_stopping])


if grad:
    print("predict train")
    train_predict = model.predict((x_dirty, y_dirty))
    print("predict test")
    test_predict = model.predict((x_dirty_test, y_dirty_test))
else:
    print("predict train")
    train_predict = model.predict(x_dirty)
    print("predict test")
    test_predict = model.predict(x_dirty_test)

print("saving")
np.save(project_folder + f"data/processed/{data}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
np.save(project_folder + f"data/processed/{data}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)
# pickle.dump(history.history, open(project_folder + f"results/{data}/history_{network}_{ISNR}dB" + postfix + ".pkl", "wb"))


# print("it took:",  (time.time()-st)/60, "mins")
