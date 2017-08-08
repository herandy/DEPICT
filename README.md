Implementation of the DEPICT algorithm.

Released as part of ECCV 2017 paper submission "Deep Clustering via Joint Convolutional Autoencoder Embedding and Relative Entropy Minimization".

Package versions used when preparing the paper:

    Theano 0.9.0
    Lasagne 0.2.dev1
    CUDA toolkit 8.0, CUDNN 5105
    Python 2.7.13 & 3.6.1

Use DEPICT.py and change its arguments accordingly to run the algorithm, you only need to specify dataset names.

The model saves the parameters at each step and reloads them if available by default, set --continue_training=False to avoid this.

To reproduce timings the MATLAB version of AC-MPI should be used, please install the MATLAB engine for Python, you can follow the original instructions for installation at:
https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html