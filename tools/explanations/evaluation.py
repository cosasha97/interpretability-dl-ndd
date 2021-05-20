import torch


def get_exp_sens(X, model, expl,exp, yy, pdt, sg_r,sg_N,sen_r,sen_N,norm,binary_I,given_expl):
    max_diff = -math.inf
    for _ in range(sen_N):
        sample = torch.FloatTensor(sample_eps_Inf(X.cpu().numpy(), sen_r, 1)).cuda()
        X_noisy = X + sample
        expl_eps, _ = get_explanation_pdt(X_noisy, model, yy, exp, sg_r, sg_N,
                                     given_expl=given_expl, binary_I=binary_I)
        max_diff = max(max_diff, np.linalg.norm(expl-expl_eps)) / norm
    return max_diff


def max_sensitivity(X, model, N, alpha=0.5):
    """
    Max-Sensitivity measures the reliability in terms of the maximum change in an explanation.
    Args:
        X: tensor, brain image
        model: pytorch model
        N: number of iterations
    """
    for _ in range(N):
        noisy_X = X + X.data.new(X.size()).normal_(0, alpha * X.std().item())
        x = torch.max(noisy_X, X.min().expand_as(X))
        noisy_X = torch.min(noisy_X, X.max().expand_as(X))