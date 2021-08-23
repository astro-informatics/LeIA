
from src.network import *
import numpy as np
import tensorflow as tf
import pickle 
import os
import time
import sys 

from src.sampling.uv_sampling import spider_sampling

ISNR = int(sys.argv[1])
network = sys.argv[2]
data = sys.argv[3]
activation = sys.argv[4]
load_weights = bool(int(sys.argv[5]))
grad = bool(int(sys.argv[6]))
set_size = int(sys.argv[7])

i = 1
# grad = False # 40x slower (27x)
if grad:
    postfix = "_" + activation + "_grad_new_" + str(i)
else:
    postfix = "_" + activation + "_new_" + str(i)


project_folder = os.environ["HOME"] + "/src_aiai/"
data_folder = project_folder + f"data/intermediate/{data}"


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


x_true, x_dirty, y_dirty, x_true_test, x_dirty_test, y_dirty_test = load_data(data_folder)

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

latest = tf.train.latest_checkpoint(checkpoint_folder)
model.load_weights(latest)


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
