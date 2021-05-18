import optimusprimal.linear_operators as linear_operators

class dictionary_tf(linear_operators.dictionary):
    def __init__(self, wav, levels, shape, axes=None):
        super().__init__(wav, levels, shape, axes=axes)

    def dir_op(self, x):
        x = x[0,0] #.numpy()
        return super().dir_op(x)


def wavelet_basis(shape, wavelets=None, levels=None, TF=False):
    if wavelets is None:
        wav = ["db8", "db6", "db4", "db2", "db1", "dirac"]
        levels = 3 #wavelet levels, makes no difference for dirac
        # you can choose the convergence criteria of the algorithm

    # if TF:
    #     return dictionary_tf(wav, levels, shape)
    return linear_operators.dictionary(wav, levels, shape)

