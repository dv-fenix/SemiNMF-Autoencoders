from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SAD(nn.Module):
  def __init__(self, num_bands: int=156):
    super(SAD, self).__init__()
    self.num_bands = num_bands

  def forward(self, input, target):
    """Spectral Angle Distance Objective
    Implementation based on the mathematical formulation presented in 'https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7061924'
    
    Params:
        input -> Output of the autoencoder corresponding to subsampled input
                tensor shape: (batch_size, num_bands)
        target -> Subsampled input Hyperspectral image (batch_size, num_bands)
        
    Returns:
        angle: SAD between input and target
    """
    try:
      input_norm = torch.sqrt(torch.bmm(input.view(-1, 1, self.num_bands), input.view(-1, self.num_bands, 1)))
      target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1)))
      
      summation = torch.bmm(input.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
      angle = torch.acos(summation/(input_norm * target_norm))
      
    
    except ValueError:
      return 0.0
    
    return angle

class SID(nn.Module):
  def __init__(self, epsilon: float=1e5):
    super(SID, self).__init__()
    self.eps = epsilon

  def forward(self, input, target):
    """Spectral Information Divergence Objective
    Note: Implementation seems unstable (epsilon required is too high)
    Implementation based on the mathematical formulation presented in 'https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7061924'
    
    Params:
        input -> Output of the autoencoder corresponding to subsampled input
                tensor shape: (batch_size, num_bands)
        target -> Subsampled input Hyperspectral image (batch_size, num_bands)
        
    Returns:
        sid: SID between input and target
    """
    normalize_inp = (input/torch.sum(input, dim=0)) + self.eps
    normalize_tar = (target/torch.sum(target, dim=0)) + self.eps
    sid = torch.sum(normalize_inp * torch.log(normalize_inp / normalize_tar) + normalize_tar * torch.log(normalize_tar / normalize_inp))
    
    return sid

class KKT(nn.Module):
  def __init__(self, W, b):
    super(KKT, self).__init__()
    self.W = W
    self.b = b

  def forward(self, model, X):
    psi = torch.linalg.pinv(self.W)
    alpha_a = torch.mm(psi, X.T)
    alpha_b = torch.mv(psi, self.b).view(3, -1)
    alpha__ = alpha_a - alpha_b
    ones = torch.ones_like(model.encoder.hidden_1.bias)
    alpha = (torch.matmul(ones, alpha__) - 1).view(1, -1)

    V_t_a = torch.inverse(torch.mm(self.W.T, self.W))
    V_t = 0.5 * torch.mv(V_t_a, ones).view(1, -1)
    lm_den = torch.mv(V_t, ones.T)
    lm_ = alpha / lm_den
    one = ones.view(1, -1)
    lm = torch.mm(lm_.T, one)

    diff_a = model.encoder.hidden_1.weight - psi
    enc_corr = torch.norm(diff_a)
    bias = model.encoder.hidden_1.bias.clone()
    bias = bias.view(3, -1)
    diff_b =  bias + alpha_b + torch.mm(V_t_a, lm.T)
    bias_corr = torch.sqrt(torch.sum((diff_b)**2))

    corr_loss = enc_corr + bias_corr
    self.W = model.decoder.weight
    self.b = model.decoder.bias
    return corr_loss