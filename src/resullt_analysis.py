# __requires__= 'numpy>=1.16.5'
# import pkg_resources
# pkg_resources.require('numpy>=1.16.5')
import numpy as np
import sys
import pickle
import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.visualisation import compare

from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error


# data = "COCO"
data = sys.argv[1]

ISNR = 30

print("load train")
x_true = np.squeeze(np.load(f"./data/intermediate/{data}/x_true_train_{ISNR}dB.npy"))
print("load test")
x_true_test = np.squeeze(np.load(f"./data/intermediate/{data}/x_true_test_{ISNR}dB.npy"))

print("define datasets")
name_net_post = [
    ("Adjoint", "adjoint", "_sigmoid"),
#     ("Learned Adjoint", "adjoint", "_sigmoid_learned_adjoint"),

    # ("U-net", "unet", "_sigmoid"),
#     ("U-net learned adjoint", "unet", "_sigmoid_learned_adjoint"),
    
    # ("dU-net", "dunet", "_sigmoid"),
#     ("dU-net learned adjoint", "dunet", "_sigmoid_learned_adjoint"),
#     ("dU-net learned grad", "dunet", "_sigmoid_learned_grad"),
#     ("dU-net grad upsample", "dunet", "_sigmoid_upsample_grad"),
#     ("U-net long", "unet", "_sigmoid_long"),
#     ("dU-net dirty", "dunet", "_sigmoid_upsample_grad_dirty"),
    
    ("U-net", "unet", "_sigmoid_same"),
    ("GU-net", "dunet", "_sigmoid_same"),
    ("HL-net sigmoid", "highlow", "_sigmoid_2"),
    ("HL-net linear", "highlow", "_linear_2"),
    ("HL-net deep", "highlow", "_linear_deep"),
    ("HL-net ramp", "highlow", "_linear_ramp"),


    # ("GU-net sigmoid 2", "dunet", "_sigmoid_same2"),
    # ("GU-net linear 2", "dunet", "_linear_same2"),
#     ("U-net small", "unet", "_sigmoid_small"),
#     ("dU-net small", "dunet", "_sigmoid_small"),
#     ("U-net smaller", "unet", "_sigmoid_smaller"),
#     ("dU-net smaller", "dunet", "_sigmoid_smaller"),
    # ("Adjoint", "adjoint", "_sigmoid"),
# 
    # ("U-net", "unet", "_linear_report"),
    # ("GU-net", "dunet", "_linear_report"),
]


results = []
results += [(name, "Train", f"./data/processed/{data}/train_predict_{net}_{ISNR}dB{post}.npy") for name, net, post in name_net_post]
results += [(name, "Test", f"./data/processed/{data}/test_predict_{net}_{ISNR}dB{post}.npy") for name, net, post in name_net_post]
results += [("Primal Dual", mode, f"./data/processed/{data}/PD_{mode.lower()}_predict_{ISNR}dB.npy") for  mode in ["Train", "Test"]]

print(results)

metrics = [
    ("PSNR", peak_signal_noise_ratio),
    ("SSIM", structural_similarity),
    ("MSE", mean_squared_error)
]

n_examples = 5
examples = []
examples.append(("True", "Train", x_true[:n_examples]))
examples.append(("True", "Test", x_true_test[:n_examples]))
statistics = pd.DataFrame(columns=["PSNR", "SSIM", "MSE", "method", "set"])#{}

print("loading datasets")
for j in  tqdm.tqdm(range(len(results))):
# for j in tqdm.tqdm(range(1)):
    name, dset, pred_path = results[j]

    try:
        pred = np.squeeze( np.load(pred_path) )
        df = pd.DataFrame()
        for metric, f in metrics:
            if dset == "Train":
                x = x_true 
            else:
                x = x_true_test
            df[metric] = [f(x[i], pred[i]) for i in range(len(x))]
            df['Method'] = name
            df['Set'] = dset
            if statistics.empty:
                statistics = df
            else:
                statistics = statistics.append(df, ignore_index=False)
        examples.append((name, dset, pred[:n_examples]))
    except:
        pass

print("saving results")
with pd.option_context('mode.use_inf_as_na', True):
    statistics.dropna(inplace=True)
    
pickle.dump(examples, open(f"./results/{data}/examples.pkl", "wb"))
statistics.to_csv(f"./results/{data}/statistics.csv")


if data == "COCO":
    # robustness data

    results = []
    results += [(name, "Test", f"./data/processed/{data}/test_predict_{net}_robustness{post}.npy") for name, net, post in name_net_post]

    statistics = pd.DataFrame(columns=["PSNR", "SSIM", "MSE", "method", "set"])#{}

    sigmas = np.repeat(np.arange(30,10,-2.5), 200)
    x_test = np.load(f"./data/intermediate/x_true_test_robustness.npy")

    for j in  tqdm.tqdm(range(len(results))):
    # for j in tqdm.tqdm(range(1)):
        name, dset, pred_path = results[j]


        try:
            pred = np.squeeze( np.load(pred_path) )
            df = pd.DataFrame()
            for metric, f in metrics:
                x = x_test
                df[metric] = [f(x[i], pred[i]) for i in range(len(x))]
                df['Method'] = name
                df['Noise'] = sigmas
                if statistics.empty:
                    statistics = df
                else:
                    statistics = statistics.append(df, ignore_index=False)
    #         examples.append((name, dset, pred[0]))
        except:
            pass


    print("saving results")
    # with pd.option_context('mode.use_inf_as_na', True):
    #     statistics.dropna(inplace=True)

    # pickle.dump(examples, open("examples_robustness.pkl", "wb"))
    statistics.to_csv(f"{data}_statistics_robustness.csv")
