import os


def use_least_loaded_gpu(least_loaded=None):
    if least_loaded is None:
        cmd = 'nvidia-smi --query-gpu="memory.used" --format=csv'
        gpu_mem_util = os.popen(cmd).read().split("\n")[:-1]
        gpu_mem_util.pop(0)
        gpu_mem_util = [util.split(' ')[0] for util in gpu_mem_util]

        cmd = 'nvidia-smi --query-gpu="utilization.gpu" --format=csv'
        gpu_util = os.popen(cmd).read().split("\n")[:-1]
        gpu_util.pop(0)
        gpu_util = [util.split(' ')[0] for util in gpu_util]

        total_util = [int(i) + int(j) for i, j in zip(gpu_mem_util, gpu_util)]
        # total_util = gpu_util
        least_loaded = total_util.index(min(total_util))
        os.environ["THEANO_FLAGS"] = "device=cuda" + str(least_loaded)
    else:
        os.environ["THEANO_FLAGS"] = "device=cuda" + str(least_loaded)


use_least_loaded_gpu()

import argparse
from functions import *
import socket

############################## settings ##############################
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=42)
parser.add_argument('--dataset', default='USPS')
parser.add_argument('--continue_training', default=True)
parser.add_argument('--datasets_path', default='/datasets/')
parser.add_argument('--feature_map_sizes', default=[50, 50, 10])
parser.add_argument('--dropouts', default=[0.1, 0.1, 0.0])
parser.add_argument('--batch_size', default=100)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', default=4000)
parser.add_argument('--reconstruct_hyperparam', default=1.)
parser.add_argument('--cluster_hyperparam', default=1.)

parser.add_argument('--architecture_visualization_flag', default=1)
parser.add_argument('--loss_acc_plt_flag', default=1)
parser.add_argument('--verbose', default=2)
args = parser.parse_args()

############################## Logging ##############################
output_path = './results/' + os.path.basename(__file__).split('.')[0] + '/' + time.strftime("%d-%m-%Y_") + \
              time.strftime("%H:%M:%S") + '_' + args.dataset + '_' + socket.gethostname()
pyscript_name = os.path.basename(__file__)
create_result_dirs(output_path, pyscript_name)
sys.stdout = Logger(output_path)
print(args)
print('----------')
print(sys.argv)

# fixed random seeds
seed = args.seed
np.random.seed(args.seed)
rng = np.random.RandomState(seed)
theano_rng = MRG_RandomStreams(seed)
lasagne.random.set_rng(np.random.RandomState(seed))
learning_rate = args.learning_rate
dataset = args.dataset
datasets_path = args.datasets_path
dropouts = args.dropouts
feature_map_sizes = args.feature_map_sizes
num_epochs = args.num_epochs
batch_size = args.batch_size
cluster_hyperparam = args.cluster_hyperparam
reconstruct_hyperparam = args.reconstruct_hyperparam
verbose = args.verbose

############################## Load Data  ##############################
X, y = load_dataset(datasets_path + dataset)
num_clusters = len(np.unique(y))
num_samples = len(y)
dimensions = [X.shape[1], X.shape[2], X.shape[3]]
print('dataset: %s \tnum_samples: %d \tnum_clusters: %d \tdimensions: %s'
      % (dataset, num_samples, num_clusters, str(dimensions)))

feature_map_sizes[-1] = num_clusters
input_var = T.tensor4('inputs')
kernel_sizes, strides, paddings, test_batch_size = dataset_settings(dataset)
print(
    '\n... build DEPICT model...\nfeature_map_sizes: %s \tdropouts: %s \tkernel_sizes: %s \tstrides: %s \tpaddings: %s'
    % (str(feature_map_sizes), str(dropouts), str(kernel_sizes), str(strides), str(paddings)))

##############################  Build DEPICT Model  ##############################
encoder, decoder, loss_recons, loss_recons_clean = build_MdA(input_var, n_in=dimensions,
                                                             feature_map_sizes=feature_map_sizes,
                                                             dropouts=dropouts, kernel_sizes=kernel_sizes,
                                                             strides=strides,
                                                             paddings=paddings)

############################## Pre-train DEPICT Model   ##############################
print("\n...Start AutoEncoder training...")
initial_time = timeit.default_timer()
train_MdA_val(dataset, X, y, input_var, decoder, encoder, loss_recons, loss_recons_clean, num_clusters, output_path,
              batch_size=batch_size, test_batch_size=test_batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
              verbose=verbose, seed=seed, continue_training=args.continue_training)

############################## Clustering Pre-trained DEPICT Features  ##############################
y_pred, centroids = Clustering(dataset, X, y, input_var, encoder, num_clusters, output_path,
                               test_batch_size=test_batch_size, seed=seed)

############################## Train DEPICT Model  ##############################
train_RLC(dataset, X, y, input_var, decoder, encoder, loss_recons, num_clusters, y_pred, output_path,
          batch_size=batch_size, test_batch_size=test_batch_size, num_epochs=num_epochs,
          learning_rate=learning_rate, rec_mult=reconstruct_hyperparam, clus_mult=cluster_hyperparam,
          centroids=centroids, continue_training=args.continue_training)

final_time = timeit.default_timer()

print('Total time for ' + dataset + ' was: ' + str((final_time - initial_time)))
