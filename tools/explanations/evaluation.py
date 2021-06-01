import torch
import numpy as np


def max_sensitivity(X, exp_method, N=10, alpha=1):
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
    # explanations for original image
    expls = exp_method.get_explanations(X, to_cpu=True)
    for _ in range(N):
        # create noisy image
        noisy_X = X + torch.normal(0, alpha * X.std(), size=X.size(),device=torch.device('cuda:0'))
        noisy_X = torch.max(noisy_X, X.min().expand_as(X))
        noisy_X = torch.min(noisy_X, X.max().expand_as(X))
        noisy_expls = exp_method.get_explanations(noisy_X, resize=False, to_cpu=True)
        # compute differences in explanations
        for target in noisy_expls:
            diff = np.linalg.norm(noisy_expls[target] - expls[target]) / np.linalg.norm(expls[target])
            if target not in max_diff:
                max_diff[target] = diff
            else:
                max_diff[target] = max(max_diff[target], diff)
        # tensor version
        # for target in noisy_expls:
        #     diff = torch.linalg.norm(noisy_expls[target] - expls[target]).item() / torch.linalg.norm(expls[target]).item()
        #     if target not in max_diff:
        #         max_diff[target] = diff
        #     else:
        #         max_diff[target] = max(max_diff[target], diff)
    return max_diff
