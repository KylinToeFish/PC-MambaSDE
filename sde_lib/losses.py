import numpy as np
import torch

from einops import repeat, rearrange

def CNLL_(targets, predictions):
    b = targets.size(0)
    if len(predictions.size()) != 4:
        predictions = rearrange(predictions, 'b t d -> 1 b t d')
    targets = repeat(targets, 'b t -> n b t', n=predictions.size(0))
    
    targets = rearrange(targets, 'n b t -> (n b t)')
    
    predictions = rearrange(predictions, 'n b t d -> (n b t) d')

    loss = torch.nn.functional.cross_entropy(predictions, targets) * b
    acc = 100 * (predictions.argmax(-1) == targets).sum()/targets.size(0) * b
    return loss, acc

def BNLL_(targets, predictions, uint8_targets=False):

    if uint8_targets:
        targets = targets / 255.0
    point_wise_error = -1 * (targets * torch.log(predictions + 1e-12) +
                             (1 - targets) * torch.log(1 - predictions + 1e-12))
    red_axis = [i + 2 for i in range(len(targets.shape) - 2)]
    sample_wise_error = torch.sum(point_wise_error, axis=red_axis)
    return torch.mean(sample_wise_error)

def MSE_(target, predicted, mask=None):

    if mask is None:
        mask = torch.ones_like(predicted)

    return torch.sum(mask * torch.square(target - predicted).mean(0))/torch.sum(mask)

def GNLL_(targets, pred_mean, pred_variance, eps = 1e-6, mask=None, normalize_dim=False):

    epsilon = eps * torch.ones_like(pred_mean)
    pred_variance = torch.maximum(pred_variance, epsilon)

    if mask == None:
        mask = torch.ones_like(pred_mean)


    const = np.log(2 * np.pi)
    sample_dim_time_wise = mask * \
        (torch.log(pred_variance) + torch.square(pred_mean - targets) / pred_variance + const)
    sample_time_wise = 0.5 * torch.sum(sample_dim_time_wise, -1)

    if normalize_dim:
        num_dim_observed = torch.sum(mask, -1)
        sample_time_wise = sample_time_wise / num_dim_observed

    sample_wise = torch.mean(sample_time_wise, -1)
    return torch.mean(sample_wise)
