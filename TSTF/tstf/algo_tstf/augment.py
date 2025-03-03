import torch


def add_noise(tensor, noise_std=0.01):
    noise = torch.randn_like(tensor[:, :-1, :]) * noise_std
    tensor[:, :-1, :] += noise
    return tensor


def random_dropout(tensor, drop_prob=0.2):
    mask = torch.rand(tensor.size(0), tensor.size(1) - 1, 1, device=tensor.device) > drop_prob
    tensor[:, :-1, :] *= mask
    return tensor


def augment_sample(tensor):
    tensor = tensor.clone()
    if torch.rand(1) < 0.5:
        tensor = add_noise(tensor)
    if torch.rand(1) < 0.5:
        tensor = random_dropout(tensor)
    return tensor
