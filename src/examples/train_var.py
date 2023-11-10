import time
from src.util import gpu_setup
gpu_setup(level=65000)

import numpy as np
import os
import sys
import tensorflow as tf

from src.networks.UNet_var import UNet_var
from src.networks.GUNet_var import GUNet_var

from src.operators.NUFFT2D_TF import NUFFT2D_TF

from src.callbacks import PredictionTimeCallback, TimeOutCallback, CSV_logger_plus 

operator = "NUFFT_Random_var"
data = "TNG"
ISNR = 30  
# extra_postfix = "_test"
extra_postfix = "_new2"
network = "UNet_var"
# network = "GUNet_var"
debug = False

# try: 
#     postfix = "_" + str(sys.argv[1])
# except:
#     postfix = ""

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()

epochs= 1000
if debug:
    epochs=1
transfer_epochs = 100

set_size = 2000
val_size = 500
save_freq = 20


Nd = (256,256)
batch_size = 20
# batch_size = 5


project_folder = os.environ["HOME"] + "/src_aiai/"
data_folder = project_folder + f"data/intermediate/{data}/{operator}/"

uv = np.load(f"./data/intermediate/{data}/NUFFT_Random_var/uv_big.npy")
uv_test = np.load(data_folder + "/uv_original.npy")

mode = int(sys.argv[1])
if mode == 0:
    # uv = uv_test
    postfix = "_known"
elif mode == 1:
    postfix = "_general"
    # postfix = "_scheduled_20"
elif mode == 2:
    postfix = "_specific_known"
elif mode == 3:
    postfix = "_specific"
elif mode == 4:
    postfix = "_general_known"

postfix += extra_postfix

checkpoint_folder = project_folder+ f"models/{data}/{operator}/{network}_{ISNR}dB{postfix}"
checkpoint_path = checkpoint_folder + "/cp-{epoch:04d}.ckpt"

print("starting model building and compilation")
st = time.time()

class PSNRMetric(tf.keras.metrics.Metric):
    """A custom TF metric that calculates the average PSNR between images in y_true and y_pred with the maximum value of max(y_true)"""
    def __init__(self, name='PSNR', **kwargs):
        super(PSNRMetric, self).__init__(name=name, **kwargs)
        self.total_PSNR = self.add_weight(name='total_PSNR', initializer='zeros')
        self.num_samples = self.add_weight(name='num_samples', initializer='zeros')
            
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_PSNR.assign_add(self.PSNR(y_true,y_pred))
        self.num_samples.assign_add(1)
                
    def result(self):
        return self.total_PSNR/self.num_samples
    
    def PSNR(self, y_true, y_pred):
        return tf.image.psnr(y_true,y_pred, max_val=tf.math.reduce_max(y_true))

m = PSNRMetric()

if network == "UNet_var":
    model = UNet_var(
       Nd, 
       uv=uv,
       op=NUFFT2D_TF, 
       depth=4, 
       conv_layers=2,
       input_type="measurements", 
       measurement_weights=1,
       batch_size=batch_size,
       residual=True,
       metrics=[PSNRMetric()]
       )
else:
    model = GUNet_var(
       Nd, 
       uv=uv,
       op=NUFFT2D_TF, 
       depth=4, 
       conv_layers=2,
       input_type="measurements", 
       measurement_weights=np.ones(len(uv)),
       batch_size=batch_size,
       residual=True,
       metrics=[PSNRMetric()]
       )

print(f"model building and compilation DONE in {(time.time()-st)/60:.2f} mins")

class Known(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, operator="NUFFT_SPIDER", ISNR=30):
        i = 0
        try:
            operator = operator.decode('utf-8')
        except:
            pass
        x = np.load(f"./data/intermediate/{data}/{operator}/x_true_train_{ISNR}dB.npy")
        y = np.load(f"./data/intermediate/{data}/{operator}/y_dirty_train_{ISNR}dB.npy")
        z = np.load(f"./data/intermediate/{data}/{operator}/sel.npy")            
        z = np.tile(z, len(y)).reshape(len(y), -1)
        while True:
            x = np.load(f"./data/intermediate/{data}/{operator}/x_true_train_{ISNR}dB_{i:03d}.npy")
            y = np.load(f"./data/intermediate/{data}/{operator}/y_dirty_train_{ISNR}dB_{i:03d}.npy")
            yield (y, z), x
            i = (i + 1) % epochs # only a 100 presaved so reuse them

    def __new__(cls, operator, ISNR=30, epochs=100):
        # assert os.path.exists(
            # f"./data/intermediate/{data}/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=((tf.complex64, tf.bool), tf.float32),
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
        
        y = np.load(f"./data/intermediate/{data}/{operator}/y_dirty_train_{ISNR}dB_{i:03d}.npy")
        z = np.ones_like(y, dtype=bool)
        np.random.seed(seed)
        sel = np.load(f"./data/intermediate/{data}/{operator}/sel.npy")
        random_sel = np.random.permutation(sel)
        z = np.tile(random_sel, len(y)).reshape(len(y), -1)
        while True:
            x = np.load(f"./data/intermediate/{data}/{operator}/x_true_train_{ISNR}dB_{i:03d}.npy")
            y = np.load(f"./data/intermediate/{data}/{operator}/y_dirty_train_{ISNR}dB_{i:03d}.npy")
            yield (y, z), x
            i = (i + 1) % epochs # only a 100 presaved so reuse them
            
    def __new__(cls, operator, seed=36202, ISNR=30, epochs=100):
        # assert os.path.exists(
            # f"./data/intermediate/{data}/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=((tf.complex64, tf.bool), tf.float32),
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
            x = np.load(f"./data/intermediate/{data}/{operator}/x_true_train_{ISNR}dB_{i:03d}.npy")
            y = np.load(f"./data/intermediate/{data}/{operator}/y_dirty_train_{ISNR}dB_{i:03d}.npy")
            
            for j in range(0, len(y), batch_size):
                
                ###
                subsel = np.ones(len(uv))
                for i in range(len(sels)-1):
                    sel_band = sels[i].astype(int) - sels[i+1].astype(int)
                    selections = np.random.permutation(np.sum(sel_band)) < np.sum(sel_band)//2
                    subsel[sel_band.astype(bool)] = selections
                z = np.tile(subsel.astype(bool), batch_size).reshape(batch_size, -1)
                ###

                # z = np.ones_like(y[j:j+batch_size], dtype=bool)
                # z[:,np.random.permutation(z.shape[1]) < z.shape[1]//2] = False
        
        
                yield (y[j:j+batch_size], z), x[j:j+batch_size]
            i = (i + 1) % epochs # only a 100 presaved so reuse them
            
    def __new__(cls, operator, seed=36202, ISNR=30, epochs=100):
        # assert os.path.exists(
            # f"./data/intermediate/{data}/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=((tf.complex64, tf.bool), tf.float32),
            args=(epochs, seed, operator, ISNR)
        )

class Validation(tf.data.Dataset):
    """a dataset that loads pre-augmented data. """

    @staticmethod
    def _generator(epochs, seed=36202, operator="NUFFT_SPIDER", ISNR=30):
        i = 0
        try:
            operator = operator.decode('utf-8')
        except:
            pass

        np.random.seed(seed)

        x = np.load(f"./data/intermediate/{data}/{operator}/x_true_val_{ISNR}dB.npy")
        y = np.load(f"./data/intermediate/{data}/{operator}/y_dirty_val_{ISNR}dB.npy")
        z = np.load(f"./data/intermediate/{data}/{operator}/sel.npy")            
        z = np.tile(z, batch_size).reshape(batch_size, -1)
        while True:
            for j in range(0, len(y), batch_size):
                yield (y[j:j+batch_size], z), x[j:j+batch_size]
            i = (i + 1) % 99 # only a 100 presaved so reuse them
            
    def __new__(cls, operator, seed=36202, ISNR=30, epochs=99):
        # assert os.path.exists(
            # f"./data/intermediate/{data}/{operator}" ), \
            # f"Could not find pregenerated dataset for operator {operator}"
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=((tf.complex64, tf.bool), tf.float32),
            args=(epochs, seed, operator, ISNR)
        )
yzx_known = Known(operator).unbatch().batch(batch_size)
yzx_fixed = Fixed(operator, seed=4620389).unbatch().batch(batch_size)
yzx_validation = Validation(operator, seed=562093).unbatch().batch(batch_size)
yzx_varying = Varying(operator, seed=65947).unbatch().batch(batch_size)


yzx_known_transfer = Known(operator, epochs=1).unbatch().batch(batch_size)

csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix + "")


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    monitor='val_PSNR',
    mode='max',
    save_best_only='True',
    save_weights_only=True,
    save_freq='epoch'#save_freq* (set_size//batch_size)
)

    

# first fit on full operator
# model.fit(yfx, steps_per_epoch=set_size//batch_size, epochs=10, callbacks=[csv_logger, cp_callback], validation_data=val_yz)

# redefine logger to append loss
# csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix + "", append=True)

# then fit on 50% subsampled operators
if mode == 0:
    print("fitting to known test sampling distribution")
    model.fit(yzx_known, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)
#     model.fit(yzx_fixed, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback]) #, validation_data=yzx_validation, validation_steps=val_size//batch_size)

elif mode == 1 or mode == 4:
    print("fitting to varying sampling distribution")
    # model.fit(yzx_varying, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback])
    model.fit(yzx_varying, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)
    
    # transfer learning:
    if mode == 4:
        epochs = transfer_epochs
        if debug:
            epochs=1
         # model = model.rebuild_with_op(uv_test)
        
        csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix + "", append=True)
        print("fitting to known test sampling distribution")
        model.fit(yzx_known_transfer, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)

elif mode == 2 or mode == 3:
    print("fitting to known random sampling distribution")
    model.fit(yzx_fixed, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)

    csv_logger = CSV_logger_plus(project_folder + f"logs/{data}/{operator}/log_{network}_{ISNR}dB" + postfix + "", append=True)

    # transfer learning:
    if mode == 2:
        epochs = transfer_epochs
        if debug:
            epochs=1
        # model = model.rebuild_with_op(uv_test)

        print("fitting to known test sampling distribution")
        model.fit(yzx_known_transfer, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)
#     elif mode == 3:
#         print("fitting to varying sampling distribution")
#         model.fit(yzx_varying, steps_per_epoch=set_size//batch_size, epochs=epochs, callbacks=[csv_logger, cp_callback], validation_data=yzx_validation, validation_steps=val_size//batch_size)


# uv_test = np.load(data_folder + "/uv.npy")
y_shape = len(uv)

print("loading train and test data")
x_true = np.load(data_folder+ f"x_true_train_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty = np.load(data_folder+ f"y_dirty_train_{ISNR}dB.npy").reshape(-1,y_shape)
z = np.load(f"./data/intermediate/{data}/{operator}/sel.npy")            
sel_dirty = np.tile(z, len(y_dirty)).reshape(len(y_dirty), -1)

x_true_test = np.load(data_folder+ f"x_true_test_{ISNR}dB.npy").reshape(-1,256,256)
y_dirty_test = np.load(data_folder+ f"y_dirty_test_{ISNR}dB.npy").reshape(-1,y_shape)
sel_dirty_test = np.tile(z, len(y_dirty_test)).reshape(len(y_dirty_test), -1)

# m_op = NUFFT2D_TF()
# m_op.plan(uv_test, (Nd[0], Nd[1]), (Nd[0]*2, Nd[1]*2), (6,6))

# model = model.rebuild_with_op(uv_test)

# w_test = calc_w(uv_test)

# y_dirty *= w_test
# y_dirty_test *= w_test

latest = tf.train.latest_checkpoint(checkpoint_folder)
model.load_weights(latest)


print("predict train")
train_predict = model.predict([y_dirty, sel_dirty], batch_size=batch_size, callbacks=[])
print("predict test")
test_predict = model.predict([y_dirty_test, sel_dirty_test], batch_size=batch_size)

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
