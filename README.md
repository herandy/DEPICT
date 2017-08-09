Implementation of the DEPICT algorithm.

Released as part of ICCV 2017 paper submission "[Deep Clustering via Joint Convolutional Autoencoder Embedding and Relative Entropy Minimization](https://arxiv.org/abs/1704.06327)".

Package versions used when preparing the paper:

    Theano 0.9.0
    Lasagne 0.2.dev1
    CUDA toolkit 8.0, CUDNN 5105
    Python 2.7.13 & 3.6.1

Use DEPICT.py and change its arguments accordingly to run the algorithm, you only need to specify dataset names by passing --dataset='USPS' for example.

By default the model saves the parameters at each step but will not load them if the files are available, to make the program check for available files and load its parameters at each step pass --continue_training.

To reproduce running times and some of results, the MATLAB version of AC-MPI should be used for some datasets. Please install the MATLAB engine for Python, you can follow the original [installation instructions](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

We have included our .theanorc file to ensure consistency.
