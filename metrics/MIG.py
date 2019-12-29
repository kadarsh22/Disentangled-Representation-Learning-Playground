from disentanglement_lib.evaluation.metrics import utils
import numpy as np


def _histogram_discretize(target, num_bins=4):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
    return discretized


def compute_mig(ground_truth_data,Model,random_state,num_train,batch_size=16):

    score_dict = {}
    mus_train, ys_train = utils.generate_batch_factor_code(ground_truth_data,Model, num_train,random_state, batch_size)
#    assert mus_train.shape[1] == num_train
    mig_score = []
    for binsize in range(2,42,4):
        discretized_mus = _histogram_discretize(mus_train,num_bins= binsize)
        m = utils.discrete_mutual_info(discretized_mus, ys_train)
        assert m.shape[0] == mus_train.shape[0]
        assert m.shape[1] == ys_train.shape[0]
        # m is [num_latents, num_factors]

        entropy = utils.discrete_entropy(ys_train)
        sorted_m = np.sort(m, axis=0)[::-1]
        a = sorted_m[0, :] - sorted_m[1, :]
        a = np.delete(a, 0, 0)
        entropy = np.delete(entropy, 0, 0)
        mig= np.mean(np.divide(a,entropy))
    mig_score.append(mig)
    mig = max(mig_score)
    return mig