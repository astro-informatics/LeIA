
from src.network import *
import numpy as np
import tensorflow as tf
import pickle 

checkpoint_path = "./models/50dB/cp-{epoch:04d}.ckpt"
ISNR = 50

x_true = np.load(f"./data/intermediate/x_true_train_{ISNR}dB.npy")[:,::2, ::2]
x_dirty = np.load(f"./data/intermediate/x_dirty_train_{ISNR}dB.npy")[:,::2, ::2]

x_true_test = np.load(f"./data/intermediate/x_true_test_{ISNR}dB.npy")[:,::2, ::2]
x_dirty_test = np.load(f"./data/intermediate/x_dirty_test_{ISNR}dB.npy")[:,::2, ::2]

model = small_unet()


epochs = 10000
save_freq = 1000
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=save_freq*7)

csv_logger = tf.keras.callbacks.CSVLogger(f"./logs/log_{ISNR}dB")

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(x_dirty, 
                    x_true, 
                    epochs=epochs, 
                    validation_data=(x_dirty_test, x_true_test),
                    callbacks=[cp_callback, csv_logger, early_stopping])

train_predict = model.predict(x_dirty)
test_predict = model.predict(x_dirty_test)

pickle.dump(history.history, open(f"./results/history_{ISNR}dB.pkl", "wb"))
np.save(f"./data/processed/train_predict_{ISNR}dB.npy", train_predict)
np.save(f"./data/processed/test_predict_{ISNR}dB.npy", test_predict)

