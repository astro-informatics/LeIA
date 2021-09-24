
from src.network import Unet
import numpy as np
import tensorflow as tf
import pickle 
import os
import time
import sys 
from src.sampling.uv_sampling import spider_sampling
from src.callbacks import PredictionTimeCallback
from util import gpu_setup
gpu_setup()



ISNR = int(sys.argv[1])
network = sys.argv[2] # adjoint, unet, dunet
activation = sys.argv[3]
load_weights = bool(int(sys.argv[4]))
learned_adjoint = bool(int(sys.argv[5]))
learned_grad = bool(int(sys.argv[6]))
grad_on_upsample = bool(int(sys.argv[7]))

i = 0
data = "COCO"
data = "GZOO"
# data = "LLPS"



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

def load_data(data_folder, ISNR=30):
    """load the pre-computed train and test data"""
    fft = lambda x: np.fft.fftshift(np.fft.fft2(
        np.fft.fftshift(x, axes=(1,2)), axes=(1,2), norm='ortho'), axes=(1,2))

    print("Loading train data")
    x_true = np.load(data_folder+ f"/x_true_train_{ISNR}dB.npy")
    x_dirty = np.load(data_folder+ f"/x_dirty_train_{ISNR}dB.npy")
    y_dirty = np.load(project_folder + f"./data/intermediate/{data}/y_dirty_train_{ISNR}dB.npy").reshape( -1,4440,1)
    
    # print("Creating fft grid train")
    # y_dirty = fft(x_dirty)

    print("Loading test data")
    x_true_test = np.load( data_folder+ f"/x_true_test_{ISNR}dB.npy")
    x_dirty_test = np.load( data_folder+ f"/x_dirty_test_{ISNR}dB.npy")
#     y_dirty_test = np.load(project_folder + f"./data/intermediate/{data}/y_dirty_test_{ISNR}dB.npy").reshape( -1,4440,1)
    y_dirty_test = np.load(project_folder + "/test.npy").reshape( -1,4440,1)
    
    # print("Creating fft grid test")
    # y_dirty_test = fft(x_dirty_test)

    return x_true, x_dirty, y_dirty, x_true_test, x_dirty_test, y_dirty_test


# x_true, x_dirty, y_dirty, x_true_test, x_dirty_test, y_dirty_test = load_data(data_folder)
y_dirty = np.load(project_folder + f"./data/intermediate/{data}/y_dirty_train_{ISNR}dB.npy").reshape( -1,4440)
y_dirty_test = np.load(project_folder + f"./data/intermediate/{data}/y_dirty_test_{ISNR}dB.npy").reshape( -1,4440)
# y_dirty_test = np.load(project_folder + "/test.npy").reshape( -1,4440)

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
if network != "adjoint" or learned_adjoint:
    latest = tf.train.latest_checkpoint(checkpoint_folder)
    model.load_weights(latest)


batch_size = 20
pt_callback = PredictionTimeCallback(project_folder + f"/results/{data}/summary.csv", batch_size) 
    
print("predict train")
train_predict = model.predict(y_dirty, batch_size=batch_size, callbacks=[pt_callback])
print("predict test")
test_predict = model.predict(y_dirty_test, batch_size=batch_size)

# y_dirty_test_robust = np.load(project_folder + "/data/intermediate/y_dirty_test_robustness.npy")
# print("predict")
# test_predict_robust = model.predict(y_dirty_test_robust, batch_size=20)
# np.save(project_folder + f"data/processed/{data}/test_predict_{network}_robustness" + postfix + ".npy", test_predict_robust)

print("saving")
np.save(project_folder + f"data/processed/{data}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
np.save(project_folder + f"data/processed/{data}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)
#pickle.dump(history.history, open(project_folder + f"results/{data}/history_{network}_{ISNR}dB" + postfix + ".pkl", "wb"))


# print("it took:",  (time.time()-st)/60, "mins")
