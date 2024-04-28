import torch
import torch.nn.functional as F

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)
    
def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)

def combine_vectors(x, y):
    return torch.cat((x.float(), y.float()), dim=1)