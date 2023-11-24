import subprocess as sp
import tensorflow as tf
import numpy as np

def gpu_setup(level=16000, n_gpus=1):
    """Set up memory growth on all available GPUs."""

    def _output_to_list(x):
        return x.decode("ascii").split("\n")[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    free = [True if val > level else False for val in values]
    gpus = tf.config.list_physical_devices("GPU")
    to_use = []
    for free, gpu in zip(free, gpus):
        if free:
            tf.config.experimental.set_memory_growth(gpu, True)
            to_use.append(gpu)
            if len(to_use) == n_gpus:
                break
    try:
        tf.config.experimental.set_visible_devices(to_use, "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "actual GPUs,", len(logical_gpus), "in use.")
    except RuntimeError as e:
        print(e)


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



def calculate_statistics(x_true_train, train_predict, x_true_test, test_predict, operator, network, ISNR, postfix=""):

    import pandas as pd
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

    metrics = [
        ("PSNR", peak_signal_noise_ratio),
        ("SSIM", structural_similarity),
        ("MSE", mean_squared_error)
    ]

    statistics = pd.DataFrame(columns=["PSNR", "SSIM", "MSE", "method", "set"])
    name = f"{network} {operator} {postfix[1:]}"

    for dset, x, pred in [("train", x_true_train, train_predict), ("test", x_true_test, test_predict)]:
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

    with pd.option_context('mode.use_inf_as_na', True):
        statistics.dropna(inplace=True)

    return statistics