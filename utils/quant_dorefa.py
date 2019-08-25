import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def uniform_quantize(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = float(2 ** k - 1)
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply


class weight_quantize_fn(nn.Module):
  def __init__(self, w_bit):
    super(weight_quantize_fn, self).__init__()
    assert w_bit <= 8 or w_bit == 32
    self.w_bit = w_bit
    self.uniform_q = uniform_quantize(k=w_bit)

  def forward(self, x):
    if self.w_bit == 32:
      weight_q = x
    elif self.w_bit == 1:
      E = torch.mean(torch.abs(x)).detach()
      weight_q = self.uniform_q(x / E) * E
    else:
      weight = torch.tanh(x)
      weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
      weight_q = 2 * self.uniform_q(weight) - 1
    return weight_q


class activation_quantize_fn(nn.Module):
  def __init__(self, a_bit):
    super(activation_quantize_fn, self).__init__()
    assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize(k=a_bit)

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q


class MYBN(nn.Module):
    '''custom implement batch normalization with autograd by Antinomy
    '''

    def __init__(self, num_features):
        super(MYBN, self).__init__()
        # auxiliary parameters
        self.num_features = num_features
        self.eps = 1e-5
        self.momentum = 0.1
        # hyper paramaters
        self.gamma = nn.Parameter(torch.Tensor(self.num_features), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(self.num_features), requires_grad=True)      
        # moving_averge
        self.moving_mean = torch.zeros(self.num_features)
        self.moving_var = torch.ones(self.num_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.ones_(self.moving_var)
        nn.init.zeros_(self.moving_mean)

    def forward(self, X):
        assert len(X.shape) in (2, 4)
        if X.device.type != 'cpu':
            self.moving_mean = self.moving_mean.cuda()
            self.moving_var = self.moving_var.cuda()
        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta,
                                         self.moving_mean, self.moving_var,
                                         self.training, self.eps, self.momentum)
        return Y


def batch_norm(X, gamma, beta, moving_mean, moving_var, is_training=True, eps=1e-5, momentum=0.9,):

    if len(X.shape) == 2:
        mu = torch.mean(X, dim=0)
        var = torch.mean((X - mu) ** 2, dim=0)
        if is_training:
            X_hat = (X - mu) / torch.sqrt(var + eps)
            moving_mean = momentum * moving_mean + (1.0 - momentum) * mu
            moving_var = momentum * moving_var + (1.0 - momentum) * var
        else:
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        out = gamma * X_hat + beta

    elif len(X.shape) == 4:
        shape_2d = (1, X.shape[1], 1, 1)
        c_max = torch.max(torch.max(torch.max(X,dim=0).values,dim=-1).values,dim=-1).values
        c_min = torch.min(torch.min(torch.min(X,dim=0).values,dim=-1).values,dim=-1).values

        mu = ((c_max+c_min)/2).view(shape_2d)
        var = (c_max - c_min).view(shape_2d) # biased
        X_hat = (X - mu) / torch.sqrt(var + eps)
        if is_training:
            X_hat = (X - mu) / torch.sqrt(var + eps)
            moving_mean = momentum * moving_mean.view(shape_2d) + (1.0 - momentum) * mu
            moving_var = momentum * moving_var.view(shape_2d) + (1.0 - momentum) * var
        else:
            X_hat = (X - moving_mean.view(shape_2d)) / torch.sqrt(moving_var.view(shape_2d) + eps)

        out = gamma.view(shape_2d) * X_hat + beta.view(shape_2d)

    return out, moving_mean, moving_var
def conv2d_Q_fold_bn(w_bit):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
        self.w_bit = w_bit
        self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, input, order=None):
        if self.training:

            weight_q = self.quantize_fn(self.weight)
            # print(np.unique(weight_q.detach().numpy()))
            x = F.conv2d(input, weight_q, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
            x = self.bn(x)
            return x
        else:
            shape_2d = [self.weight.shape[0],1,1,1]
            gamma = self.bn.weight.view(shape_2d)
            beta = self.bn.bias.view(shape_2d)
            mv = self.bn.running_var.view(shape_2d)
            mm = self.bn.running_mean.view(shape_2d)

            w = self.weight*gamma/torch.sqrt(mv)
            if self.bias:
                b = self.bias.view(shape_2d)
                b = beta-gamma/torch.sqrt(mv)*(mm-b)
            else:
                b = beta-gamma/torch.sqrt(mv)*mm              

            weight_q = self.quantize_fn(w)
            x = F.conv2d(input, weight_q, torch.squeeze(b), self.stride,
                          self.padding, self.dilation, self.groups)
            return x


  return Conv2d_Q

def conv2d_Q_fn(w_bit):
  class Conv2d_Q(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
      super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilation, groups, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input, order=None):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.conv2d(input, weight_q, self.bias, self.stride,
                      self.padding, self.dilation, self.groups)

  return Conv2d_Q


def linear_Q_fn(w_bit):
  class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
      super(Linear_Q, self).__init__(in_features, out_features, bias)
      self.w_bit = w_bit
      self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

    def forward(self, input):
      weight_q = self.quantize_fn(self.weight)
      # print(np.unique(weight_q.detach().numpy()))
      return F.linear(input, weight_q, self.bias)

  return Linear_Q


if __name__ == '__main__':
  import numpy as np
  import matplotlib.pyplot as plt

  a = torch.rand(1, 3, 32, 32)

  Conv2d = conv2d_Q_fn(w_bit=2)
  conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
  act = activation_quantize_fn(a_bit=3)

  b = conv(a)
  b.retain_grad()
  c = act(b)
  d = torch.mean(c)
  d.retain_grad()

  d.backward()
  pass
