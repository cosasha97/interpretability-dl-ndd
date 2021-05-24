import torch
import numpy as np


def max_sensitivity(X, exp_method, N=1000, alpha=0.5):
    """
    Max-Sensitivity measures the reliability in terms of the maximum change in an explanation.
    Args:
        X: tensor, brain image
        exp_method: explanation method. Must have a get_explanations(self, input_image) attribute function
            which takes an image as input and returns a dictionary mapping branches to explanation maps
        N: number of iterations
        alpha: float, multiplicative parameter of the std of the gaussian law used to sample noise
    """
    if torch.cuda.is_available():
        X = X.cuda()

    max_diff = dict()
    expls = exp_method.get_explanations(X)
    for _ in range(N):
        # create noisy image
        noisy_X = X + X.data.new(X.size()).normal_(0, alpha * X.std().item()).cuda()
        noisy_X = torch.max(noisy_X, X.min().expand_as(X))
        noisy_X = torch.min(noisy_X, X.max().expand_as(X))
        noisy_expls = exp_method.get_explanations(noisy_X)
        # compute differences in explanations
        for target in noisy_expls:
            diff = np.linalg.norm(noisy_expls[target] - expls[target])/np.linalg.norm(expls[target])
            if target not in max_diff:
                max_diff[target] = diff
            else:
                max_diff[target] = max(max_diff[target], diff)
    return max_diff
