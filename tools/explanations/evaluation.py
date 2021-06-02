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
    print('hello')
    # explanations for original image
    expls = exp_method.get_explanations(X, resize=False, to_cpu=False)
    for _ in range(N):
        # create noisy image
        noisy_X = X + torch.normal(0, alpha * X.std(), size=X.size(), device=torch.device('cuda:0'))
        noisy_X = torch.max(noisy_X, X.min().expand_as(X))
        noisy_X = torch.min(noisy_X, X.max().expand_as(X))
        noisy_expls = exp_method.get_explanations(noisy_X, resize=False, to_cpu=False)
        # compute differences in explanations
        for target in noisy_expls:
            diff = torch.linalg.norm(noisy_expls[target] - expls[target]) / torch.linalg.norm(expls[target])
            if target not in max_diff:
                max_diff[target] = diff
            else:
                max_diff[target] = max(max_diff[target], diff)
    return max_diff

# def max_sensitivity(X, exp_method, N=10, alpha=1):
#     """
#     Max-Sensitivity measures the reliability in terms of the maximum change in an explanation.
#     Args:
#         X: tensor, brain image
#         exp_method: explanation method. Must have a get_explanations(self, input_image) attribute function
#             which takes an image as input and returns a dictionary mapping branches to explanation maps
#         N: number of iterations
#         alpha: float, multiplicative parameter of the std of the gaussian law used to sample noise
#     """
#     if torch.cuda.is_available():
#         X = X.cuda()
#
#     max_diff = dict()
#     # explanations for original image
#     expls = exp_method.get_explanations(X, to_cpu=True)
#     for _ in range(N):
#         # create noisy image
#         noisy_X = X + torch.normal(0, alpha * X.std(), size=X.size(),device=torch.device('cuda:0'))
#         noisy_X = torch.max(noisy_X, X.min().expand_as(X))
#         noisy_X = torch.min(noisy_X, X.max().expand_as(X))
#         noisy_expls = exp_method.get_explanations(noisy_X, resize=False, to_cpu=True)
#         # compute differences in explanations
#         for target in noisy_expls:
#             diff = np.linalg.norm(noisy_expls[target] - expls[target]) / np.linalg.norm(expls[target])
#             if target not in max_diff:
#                 max_diff[target] = diff
#             else:
#                 max_diff[target] = max(max_diff[target], diff)
#         # tensor version
#         # for target in noisy_expls:
#         #     diff = torch.linalg.norm(noisy_expls[target] - expls[target]).item() / torch.linalg.norm(expls[target]).item()
#         #     if target not in max_diff:
#         #         max_diff[target] = diff
#         #     else:
#         #         max_diff[target] = max(max_diff[target], diff)
#     return max_diff


def MoRF(X, model, exp_method, K=None, group_size=20000, AUC=False, batch_size=16, to_cuda=False):
    """
    Most relevant first: measures the reliability of an explanation by testing
    how fast the output decreases, while we progressively remove information (e.g., perturb pixels)
    from the input 𝑥𝑥 (e.g., image), that appears as the most relevant by the explanation.
    Args:
        X: tensor, brain image, with shape (1, n_channels, depth, height, width). The two first dimensions
            are optional.
        exp_method: explanation method. Must have a get_explanations(self, input_image) attribute function
            which takes an image as input and returns a dictionary mapping branches to explanation maps
        K: number of group of relevant pixels to remove
        group_size: int, size of a group of pixels to remove
        AUC: bool. If True: compute and return area under the curve obtained after removing successively
            the K most relevant pixels.
        batch_size: int, number of images passed to the model each time

    TO DO:
        - add several methods to perturb pixels
    """
    if to_cuda and torch.cuda.is_available():
        X = X.cuda()

    if K is None:
        K = np.prod(X.shape) // 8

    # reshpae X if necessary
    while len(X.shape) < 5:
        X = X[None, ...]

    # original predictions
    preds = model(X)
    # explanations for original image
    expls = exp_method.get_explanations(X, resize=True)
    # explanations for new images
    new_preds = dict()

    # def update_dict

    for target in expls:
        # Indices of the sorted elements of the explanations:
        ids = np.unravel_index(np.argsort(-expls[target], axis=None), expls[target].shape)

        if AUC:
            # number of
            removed_pixels = 0
            while removed_pixels < K:
                # create batch of images
                bs = min(batch_size, (K - removed_pixels) % group_size)
                batch_X = torch.tile(X, (bs, 1, 1, 1, 1))
                for k in range(1, batch_size):
                    index = k * group_size
                    batch_X[k, 0, ids[0][:index], ids[1][:index], ids[2][:index]] = 0
                    new_preds[target] = model(batch_X)
        else:
            # compute MoRF removing the K most relevant pixels
            batch_X = X.copy()
            batch_X[0, 0, ids[0][:K], ids[1][:K], ids[2][:K]] = 0
            new_preds[target] = model(batch_X)