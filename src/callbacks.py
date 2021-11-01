import csv
import time
import numpy as np
import tensorflow as tf


class TimeOutCallback(tf.keras.callbacks.Callback):
    """Based on https://stackoverflow.com/questions/58096219/save-a-tensorflow-model-after-a-fixed-training-time"""
    def __init__(self, timeout, checkpoint_path):
        super().__init__()
        self.timeout = timeout  # time in minutes
        self.checkpoint_path = checkpoint_path

    def on_train_begin(self, logs=None):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if time.time() - self.t0 > self.timeout * 60: 
            print(f"\nReached {(time.time() - self.t0) / 60:.3f} minutes of training, stopping")
            self.model.stop_training = True
            self.model.save_weights(self.checkpoint_path.format(epoch=epoch+1))
            
class CSV_logger_plus(tf.keras.callbacks.CSVLogger):
    """A CSV logger that logs the time since start of training as well as the epochs"""
    def on_train_begin(self, logs=None):
        self.t0 = time.time() # start time
        super().on_train_begin(logs)
    
    def on_epoch_end(self, epoch, logs=None):
        logs['time'] = time.time() - self.t0
        super().on_epoch_end(epoch, logs)
        
        
        

class PredictionTimeCallback(tf.keras.callbacks.Callback):    
    def __init__(self, filename, batch_size=1):
        self.csv_file = open(filename, "w")
        self.begin_timings = []
        self.end_timings = []
        self.batch_size = batch_size # for outputting time per sample

    def on_predict_batch_begin(self, batch, logs=None):
        self.begin_timings.append(time.time())

    def on_predict_batch_end(self, batch, logs=None):
        self.end_timings.append(time.time())

    def on_predict_end(self, logs=None):        
        if hasattr(self.model, '_collected_trainable_weights'):
            trainable_count = self.count_params(self.model._collected_trainable_weights)
        else:
            trainable_count = self.count_params(self.model.trainable_weights)

        non_trainable_count = self.count_params(self.model.non_trainable_weights)

        self.timings = np.array(self.end_timings) - np.array(self.begin_timings)
        self.csv_file.write("Mean time,{}\n".format(np.mean(self.timings/self.batch_size)))
        self.csv_file.write("Std time,{}\n".format(np.std(self.timings/self.batch_size)))
        self.csv_file.write('Total params,{}\n'.format(trainable_count + non_trainable_count))
        self.csv_file.write('Trainable params,{}\n'.format(trainable_count))
        self.csv_file.write('Non-trainable params,{}\n'.format(non_trainable_count))
        self.csv_file.close()
    
    @staticmethod
    def count_params(weights):
        """Count the total number of scalars composing the weights.
        Args:
          weights: An iterable containing the weights on which to compute params
        Returns:
          The total number of scalars composing the weights
        
        (from tf.keras.utils.layer_utils.py)
        """
        unique_weights = {id(w): w for w in weights}.values()
        # Ignore TrackableWeightHandlers, which will not have a shape defined.
        unique_weights = [w for w in unique_weights if hasattr(w, 'shape')]
        weight_shapes = [w.shape.as_list() for w in unique_weights]
        standardized_weight_shapes = [
          [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
        ]
        return int(sum(np.prod(p) for p in standardized_weight_shapes))

    
class CSV_logger_plus(tf.keras.callbacks.CSVLogger):
    def on_train_begin(self, logs=None):
        self.t0 = time.time() # start time
        super().on_train_begin(logs)
    
    def on_epoch_end(self, epoch, logs=None):
        logs['time'] = time.time() - self.t0
        super().on_epoch_end(epoch, logs)