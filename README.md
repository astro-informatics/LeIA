# src_aiai

most of the package can be found in the `src/` directory. The most important files are:


- operators/measurement.py  # contains the Non-Uniform FFT operators both 1D and 2D, implemented in both TensorFlow and using SciPy
- sampling/uv_sampling.py   # contains different uv sampling distributions including that of the SPIDER instrument
- solvers/solvers.py        # contains some reconstruction algorithms based on Primal Dual with  OptimusPrimal

- network.py                # Contains the main network architecture as well as the gradient operator
- train_network.py          # Used for the training of networks
- predict_network.py        # Used for prediction using saved networks
- dataset.py                # Setup for the TensorFlow dataset, including methods for data augmentation
- process_PD.py             # Used for reconstructing the images using Primal Dual



The `notebooks/` folder contains a lot of notebooks with some examples of the code used as well as visualisations of the results
