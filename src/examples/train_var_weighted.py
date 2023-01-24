import time
from src.util import gpu_setup
gpu_setup()

import numpy as np
import os
import sys
import tensorflow as tf

from src.networks.UNet_var import UNet_var
from src.networks.GUNet_var_weighted import GUNet_var

from src.operators.NUFFT2D_TF import NUFFT2D_TF

from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 

operator = "NUFFT_Random_var"
data = "COCO"
ISNR = 30  
extra_postfix = "_weighted_exclusion2"
# extra_postfix = "_weighted"
# network = "UNet_var"
network = "GUNet_var"


# try: 
#     postfix = "_" + str(sys.argv[1])
# except:
#     postfix = ""

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

epochs= 10#200 
set_size = 2000
save_freq = 20


Nd = (256,256)
# batch_size = 20
batch_size = 5


project_folder = os.environ["HOME"] + "/src_aiai/"
data_folder = project_folder + f"data/intermediate/{data}/{operator}/"

uv = np.load("./data/intermediate/COCO/NUFFT_Random_var/uv_big.npy")
uv_test = np.load(data_folder + "/uv_original.npy")

mode = int(sys.argv[1])
if mode == 0:
    uv = uv_test
    postfix = "_known"
elif mode == 1:
    postfix = "_generalized"
    # postfix = "_scheduled_20"
elif mode == 2:
    postfix = "_transfer_known"
elif mode == 3:
    postfix = "_transfer_generalized"

postfix += extra_postfix

checkpoint_folder = project_folder+ f"models/{data}/{operator}/{network}_{ISNR}dB{postfix}"
checkpoint_path = checkpoint_folder + "/cp-{epoch:04d}.ckpt"

print("starting model building and compilation")
st = time.time()

# model = UNet_var(
#     Nd, 
#     uv=uv,
#     op=NUFFT2D_TF, 
#     depth=4, 
#     conv_layers=2,
#     input_type="measurements", 
#     measurement_weights=1,
#     batch_size=batch_size,
#     residual=True
#     )


model = GUNet_var(
    Nd, 
    uv=uv,
    op=NUFFT2D_TF, 
    depth=4, 
    conv_layers=2,
    input_type="measurements", 
    measurement_weights=np.ones(len(uv)),
    batch_size=batch_size,
    residual=True
    )

print(f"model building and compilation DONE in {(time.time()-st)/60:.2f} mins")

def calc_w(uv):
    grid_cell = 2*np.pi /512 
    binned = (uv[:,:]+np.pi+.5*grid_cell) // grid_cell
    binned = [tuple(x) for x in binned]
    w_gridded = np.zeros(uv.shape[0])

    d = {}
    for i in binned:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1

    for i in range(len(w_gridded)):
        w_gridded[i] = d[binned[i]]
    # w = 
    w = 1/w_gridded
#     w /= w.max()
    return w  

class Known(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, operator="NUFFT_SPIDER", ISNR=30):
        i = 0
        try:
            operator = operator.decode('utf-8')
        except:
            pass
        x = np.load(f"./data/intermediate/COCO/{operator}/x_true_train_{ISNR}dB.npy")
        y = np.load(f"./data/intermediate/COCO/{operator}/y_dirty_train_{ISNR}dB.npy")
        z = np.ones_like(y, dtype=bool)
        w = calc_w(uv[z])
        while True:
            yield (y, z, w), x
            i = (i + 1) % 100 # only a 100 presaved so reuse them

    def __new__(cls, operator, ISNR=30, epochs=100):
        # assert os.path.exists(
            # f"./data/intermediate/COCO/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=((tf.complex64, tf.bool, tf.float32), tf.float32),
            args=(epochs, operator, ISNR)
        )
    
class Fixed(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, seed=36202, operator="NUFFT_SPIDER", ISNR=30):
        i = 0
        try:
            operator = operator.decode('utf-8')
        except:
            pass
        
        y = np.load(f"./data/intermediate/COCO/{operator}/y_dirty_train_{ISNR}dB_{i:03d}.npy")
        z = np.ones_like(y, dtype=bool)
        np.random.seed(seed)
        z[:,np.random.permutation(z.shape[1]) < z.shape[1]//2] = False
        w = np.zeros_like(y, dtype=np.float32)
        w[:, z[0]] = calc_w(uv[z[0]])

        while True:
            x = np.load(f"./data/intermediate/COCO/{operator}/x_true_train_{ISNR}dB_{i:03d}.npy")
            y = np.load(f"./data/intermediate/COCO/{operator}/y_dirty_train_{ISNR}dB_{i:03d}.npy")
            yield (y, z, w), x
            i = (i + 1) % 100 # only a 100 presaved so reuse them
            
    def __new__(cls, operator, seed=36202, ISNR=30, epochs=100):
        # assert os.path.exists(
            # f"./data/intermediate/COCO/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=((tf.complex64, tf.bool, tf.float32), tf.float32),
            args=(epochs, seed, operator, ISNR)
        )

class Varying(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, seed=36202, operator="NUFFT_SPIDER", ISNR=30):
        i = 0
        try:
            operator = operator.decode('utf-8')
        except:
            pass

        np.random.seed(seed)
        
        ###
        depth = 4
        sels = []
        for i in range(depth+1):
            sel =  np.all(uv <  np.pi / 2**i, axis=1) # square selection (with exclusion region outside)
            sels.append(sel)
        sels.append(np.zeros_like(sels[0]))
        ###

        
        while True:
            x = np.load(f"./data/intermediate/COCO/{operator}/x_true_train_{ISNR}dB_{i:03d}.npy")
            y = np.load(f"./data/intermediate/COCO/{operator}/y_dirty_train_{ISNR}dB_{i:03d}.npy")
            
            for j in range(0, len(y), batch_size):
                
                ###
                subsel = np.ones(len(uv))
                for i in range(len(sels)-1):
                    sel_band = sels[i].astype(int) - sels[i+1].astype(int)
                    if i == len(sels)-2:
                        sel_band *= np.linalg.norm(uv, axis=1) > 4*np.pi/256
                    selections = np.random.permutation(np.sum(sel_band)) < np.sum(sel_band)//2
                    subsel[sel_band.astype(bool)] = selections
#                 w = np.ones_like(y)
                w = np.zeros_like(y, dtype=np.float32)
                w[:,subsel.astype(bool)] = calc_w(uv[subsel.astype(bool)])
                z = np.tile(subsel.astype(bool), batch_size).reshape(batch_size, -1)

                ###

                # z = np.ones_like(y[j:j+batch_size], dtype=bool)
                # z[:,np.random.permutation(z.shape[1]) < z.shape[1]//2] = False
        
        
                yield (y[j:j+batch_size], z, w[j:j+batch_size]), x[j:j+batch_size]
            i = (i + 1) % 100 # only a 100 presaved so reuse them
            
    def __new__(cls, operator, seed=36202, ISNR=30, epochs=100):
        # assert os.path.exists(
            # f"./data/intermediate/COCO/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=((tf.complex64, tf.bool, tf.float32), tf.float32),
            args=(epochs, seed, operator, ISNR)
        )

yzx_known = Known(operator).unbatch().batch(batch_size)
yzx_fixed = Fixed(operator, seed=4620389).unbatch().batch(batch_size)
yzx_validation = Fixed(operator, seed=562093).unbatch().batch(batch_size)
yzx_varying = Varying(operator, seed=65947).unbatch().batch(batch_size)

csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix + "")


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=save_freq* (set_size//batch_size)
)


# first fit on full operator
# model.fit(yfx, steps_per_epoch=set_size//batch_size, epochs=10, callbacks=[csv_logger, cp_callback], validation_data=val_yz)

# redefine logger to append loss
# csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix + "", append=True)

# then fit on 50% subsampled operators
if mode == 0:
    print("fitting to known test sampling distribution")
    # model.fit(yzx_known, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback]) #, validation_data=yzx_validation, validation_steps=set_size//batch_size)
    model.fit(yzx_fixed, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback]) #, validation_data=yzx_validation, validation_steps=set_size//batch_size)

elif mode == 1:
    print("fitting to varying sampling distribution")
    #latest = tf.train.latest_checkpoint(checkpoint_folder)
    #model.load_weights(latest)
    
    model.fit(yzx_varying, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=set_size//batch_size)
    # model.fit(yzx_varying, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback])#, validation_data=yzx_validation, validation_steps=set_size//batch_size)
    
elif mode == 2 or mode == 3:
    print("fitting to known random sampling distribution")
    model.fit(yzx_fixed, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=set_size//batch_size)

    csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix + "", append=True)

    epochs = 25
    if mode == 2:
        model = model.rebuild_with_op(uv_test)

        print("fitting to known test sampling distribution")
        model.fit(yzx_known, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback])#, validation_data=yzx_validation, validation_steps=set_size//batch_size)
    elif mode == 3:
        print("fitting to varying sampling distribution")
        model.fit(yzx_varying, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=set_size//batch_size)


uv_test = np.load(data_folder + "/uv_original.npy")
y_shape = len(uv_test)

print("loading train and test data")
x_true = np.load(data_folder+ f"x_true_train_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty = np.load(data_folder+ f"y_dirty_train_{ISNR}dB.npy").reshape(-1,y_shape)
sel_dirty = np.ones_like(y_dirty, dtype=bool)
weights = calc_w(uv_test)

x_true_test = np.load(data_folder+ f"x_true_test_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty_test = np.load(data_folder+ f"y_dirty_test_{ISNR}dB.npy").reshape(-1,y_shape)
sel_dirty_test = np.ones_like(y_dirty_test, dtype=bool)

w_dirty = np.ones_like(y_dirty)
w_dirty_test = np.ones_like(y_dirty_test)

# w = np.zeros_like(y, dtype=np.float32)
# w[:,:] = w

# m_op = NUFFT2D_TF()
# m_op.plan(uv_test, (Nd[0], Nd[1]), (Nd[0]*2, Nd[1]*2), (6,6))
batch_size = 1
model = model.rebuild_with_op(uv_test, batch_size)

w_test = calc_w(uv_test)

w_dirty *= w_test
w_dirty_test *= w_test


print("predict train")
train_predict = model.predict([y_dirty, sel_dirty, w_dirty], batch_size=batch_size, callbacks=[])
print("predict test")
test_predict = model.predict([y_dirty_test, sel_dirty_test, w_dirty_test], batch_size=batch_size)

operator = "NUFFT_Random_var"


print("saving train and test predictions")
os.makedirs(project_folder + f"data/processed/{data}/{operator}", exist_ok=True)
np.save(project_folder + f"data/processed/{data}/{operator}/train_predict_{network}_{ISNR}dB" + postfix + ".npy", train_predict)
np.save(project_folder + f"data/processed/{data}/{operator}/test_predict_{network}_{ISNR}dB" + postfix + ".npy", test_predict)


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
        stats = [f(x[i], pred[i]) for i in range(len(x))]
        df[metric] = stats
        df['Method'] = name
        df['Set'] = dset
        if statistics.empty:
            statistics = df
        else:
            statistics = statistics.append(df, ignore_index=False)
        print(name, dset, metric, np.mean(stats))
print("saving results")
with pd.option_context('mode.use_inf_as_na', True):
    statistics.dropna(inplace=True)

statistics.to_csv(project_folder + f"results/{data}/{operator}/statistics_{network}_{ISNR}dB{postfix}.csv")

