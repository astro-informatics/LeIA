
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error

imshow_kwargs = imshow_kwargs = {
#     "cmap":'afmhot',
    # "cmap":'gray',
    # "origin":'lower'
    }

def compare(images, ncols=None, nrows=None, titles=None, same_scale=False, colorbar=False, cmap='viridis'):
    """[summary]
    TODO add colorbars
    Args:
        images ([type]): [description]
        ncols ([type], optional): [description]. Defaults to None.
        nrows ([type], optional): [description]. Defaults to None.
        titles ([type], optional): [description]. Defaults to None.
        same_scale (bool, optional): [description]. Defaults to False.
    """

    if not nrows:
        nrows = 1
    if not ncols:
        ncols = 1
    if ncols*nrows < len(images):
        ncols = len(images)
        nrows = 1
    if not titles:
        titles = [""]*len(images)
    
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, 
        figsize=(ncols*6, nrows*6), squeeze=False)

    a = ax[0,0].imshow(images[0], **imshow_kwargs)    
    ax[0,0].set_title(titles[0])
    vmin, vmax = a.get_clim()
    for i in range(0, nrows):
        for j in range(ncols):
            if i +j*nrows >= len(images):
                break
            if same_scale:
                im = ax[i,j].imshow(images[j + i*ncols], 
                    **imshow_kwargs, vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                im = ax[i,j].imshow(images[j + i*ncols], cmap=cmap, **imshow_kwargs)
            ax[i,j].set_title(titles[j + i*ncols])

  
            if colorbar:
                plt.colorbar(im, ax=ax[i,j], fraction=0.046, pad=0.04)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            
#     plt.show()

def visualise_benchmark(results):
    print(f"Solved using {results['name']}")
    print(f"ISNR: {results['ISNR']}dB")
    print(f"Solving took {results['runtime']/60:.2f} mins")
    compare([
        results["x_true"], 
        results["solution"], 
        results["x_true"] -results["solution"]
        ], titles=["x_true", "solution", f"difference (RSE: {results['RSE']:.3e})"])


def print_statistics(statistics, results, metrics, latex=False):
    if latex:
        separator = " & "
        lines = ""
    else:
        separator = "|"
        lines = "|"
    

    print( f"{'Name':40}", end=separator)
    for metric, f in metrics:
        print(f"{metric:20}", end=separator)
    if latex:
        print("\\\\", end="")
    print()
    for name, set, _ in sorted(results):
        if name in statistics.Method.values:
            print(f"{name+'_'+set:40}", end=separator)
            for metric, f in metrics:
                if latex:
                    print(f"$ {np.mean(statistics[metric][(statistics.Method == name) * (statistics.Set == set)]):8.3f} \pm {np.std(statistics[metric][(statistics.Method == name) * (statistics.Set == set)]):7.3f} $", end=separator)
                else:
                    print(f"{np.mean(statistics[metric][(statistics.Method == name) * (statistics.Set == set)]):8.3f} \pm {np.std(statistics[metric][(statistics.Method == name) * (statistics.Set == set)]):7.3f}", end=separator)
                
                # median = np.median(statistics[metric][(statistics.Method == name) * (statistics.Set == set)])
                # smad = 1.4826* np.median(np.abs(statistics[metric][(statistics.Method == name) * (statistics.Set == set)] - median))
                # print(f"{median:8.3f} \pm {smad:7.3f}|", end="")
            if latex:
                print("\\\\", end="")
            print()

def plot_statistics(statistics, metrics, ylims=[[0,40], [0,1], [0, 0.04]], split=True, order=None):
    #TODO add some better ytick labels, currently to many/much precision
    fig, ax = plt.subplots(ncols=len(metrics), figsize=(len(metrics)*8, 6))
    for idx, (metric, _) in enumerate(metrics):
        plt.sca(ax[idx])
        sns.set_style('whitegrid')
        sns.violinplot(data=statistics, x='Method', y=metric, split=split, hue='Set', palette="Set3", bw=.2, cut=1, linewidth=1, inner='quart', orientation='v', order=order)
        # sns.violinplot(data=statistics, x='Method', y=metric, split=False, hue='Set', palette="Set3", bw=.2, cut=1, linewidth=1, inner='quart', orientation='v')
        plt.ylabel(metric, fontsize='x-large')
        plt.xlabel("")
        ax[idx].tick_params(labelsize='x-large', rotation=90)
        # ax[idx].xaxis.label.set_size('x-large')
        sns.despine(left=True, bottom=True)
        plt.legend(loc="upper left", fontsize='large')
    ax[0].set_ylim(ylims[0])
    ax[1].set_ylim(ylims[1])
    ax[2].set_ylim(ylims[2])
    plt.show()


def results_and_metrics(data, ISNR):
    name_net_post = [
        ("Adjoint", "adjoint", "_sigmoid"),
        ("Learned Adjoint", "adjoint", "_sigmoid_learned_adjoint"),

        ("U-net", "unet", "_sigmoid"),
        ("U-net learned adjoint", "unet", "_sigmoid_learned_adjoint"),
        
        ("dU-net", "dunet", "_sigmoid"),
        ("dU-net learned adjoint", "dunet", "_sigmoid_learned_adjoint"),
        ("dU-net learned grad", "dunet", "_sigmoid_learned_grad"),
        ("dU-net grad upsample", "dunet", "_sigmoid_upsample_grad"),
        ("dU-net dirty", "dunet", "_sigmoid_upsample_grad_dirty"),

    ]

    results = []
    results += [(name, "Train", f"./data/processed/{data}/train_predict_{net}_{ISNR}dB{post}.npy") for name, net, post in name_net_post]
    results += [(name, "Test", f"./data/processed/{data}/test_predict_{net}_{ISNR}dB{post}.npy") for name, net, post in name_net_post]
    results += [("Primal Dual", mode, f"./data/processed/{data}/PD_{mode.lower()}_predict_{ISNR}dB.npy") for  mode in ["Train", "Test"]]
    # results += [("Primal Dual", mode, f"./data/processed/{data}/PD_{mode.lower()}_predict_{ISNR}dB.npy") for  mode in ["Train"]]


    metrics = [
        ("PSNR", peak_signal_noise_ratio),
        ("SSIM", structural_similarity),
        ("MSE", mean_squared_error)
    ]
    return name_net_post, results, metrics