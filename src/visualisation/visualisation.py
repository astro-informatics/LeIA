
import matplotlib.pyplot as plt


imshow_kwargs = imshow_kwargs = {
#     "cmap":'afmhot',
    "cmap":'gray',
    # "origin":'lower'
    }

def compare(images, ncols=None, nrows=None, titles=None, same_scale=False, colorbar=False):
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
            if same_scale:
                im = ax[i,j].imshow(images[j + i*ncols], 
                    **imshow_kwargs, vmin=vmin, vmax=vmax)
            else:
                im = ax[i,j].imshow(images[j + i*ncols], **imshow_kwargs)
            ax[i,j].set_title(titles[j + i*ncols])

            if i +j*nrows >= len(images):
                break
            if colorbar:
                plt.colorbar(im, ax=ax[i,j], fraction=0.046, pad=0.04)
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            
    plt.show()

def visualise_benchmark(results):
    print(f"Solved using {results['name']}")
    print(f"ISNR: {results['ISNR']}dB")
    print(f"Solving took {results['runtime']/60:.2f} mins")
    compare([
        results["x_true"], 
        results["solution"], 
        results["x_true"] -results["solution"]
        ], titles=["x_true", "solution", f"difference (RSE: {results['RSE']:.3e})"])

