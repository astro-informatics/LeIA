
from src.network import *
import numpy as np
import tensorflow as tf
import pickle 
import os
import time
import sys 

st = time.time()

ISNR = int(sys.argv[1])
network = sys.argv[2]
data = sys.argv[3]
activation = sys.argv[4]

postfix = "_" + activation + "_grad"

project_folder = os.environ["HOME"] +"/src_aiai/"
checkpoint_path = project_folder + "models/"+ data + "/" + network +"_" + str(ISNR) + "dB" + postfix + "/cp-{epoch:04d}.ckpt"

latest = tf.train.latest_checkpoint(project_folder + "models/" + data + "/" + network + "_" + str(ISNR) + "dB" + postfix + "/")

# fft = lambda x: tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(x)))
fft = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x, axes=(-2,-1)), norm='ortho'), axes=(-2,-1))

print("Loading train data")
x_true = np.load(project_folder + f"data/intermediate/{data}/x_true_train_{ISNR}dB.npy")
x_dirty = np.load(project_folder +f"data/intermediate/{data}/x_dirty_train_{ISNR}dB.npy")
y_dirty = fft(x_dirty[:,:,:,0] +0j)[:,:,:,np.newaxis]


print("Loading test data")
x_true_test = np.load(project_folder + f"data/intermediate/{data}/x_true_test_{ISNR}dB.npy")
x_dirty_test = np.load(project_folder + f"data/intermediate/{data}/x_dirty_test_{ISNR}dB.npy")
y_dirty_test = fft(x_dirty_test[:,:,:,0] +0j)[:,:,:,np.newaxis]

# mean, std = np.mean(x_dirty), np.std(x_dirty)
# x_dirty = (x_dirty - mean)/std
# x_dirty_test = (x_dirty_test - mean)/std

if network == 'small':
    model = small_unet(
        output_activation = activation
    )
else:
    model = medium_unet(    
        output_activation = activation
    )

print("loading weights")
model.load_weights(latest)


print("predicting train")
train_predict = model.predict((x_dirty, y_dirty))

print("predicting test")
test_predict = model.predict((x_dirty_test, y_dirty_test))

#pickle.dump(history.history, open(project_folder + f"results/{data}/history_{network}_{ISNR}dB.pkl", "wb"))
np.save(project_folder + f"data/processed/{data}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
np.save(project_folder + f"data/processed/{data}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)

print("it took:",  (time.time()-st)/60, "mins")
