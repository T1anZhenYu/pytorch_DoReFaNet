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
    def __init__(self,channel,decay=0.9,eps=0.00001,affine= True):
        super(MYBN,self).__init__()
        self.channel= channel
        if affine:
            self.gamma = Variable(torch.ones(channel,dtype=torch.float),requires_grad = True).cuda()
            self.beta = Variable(torch.zeros(channel,dtype=torch.float),requires_grad = True).cuda()
        else:
            self.gamma = Variable(torch.ones(channel,dtype=torch.float),requires_grad = False).cuda()
            self.beta = Variable(torch.zeros(channel,dtype=torch.float),requires_grad = False).cuda()         
        self.moving_mean = Variable(torch.zeros(channel,dtype=torch.float),requires_grad = False).cuda()
        self.moving_var = Variable(torch.ones(channel,dtype=torch.float),requires_grad = False).cuda()    
        self.decay = decay
        self.eps = eps
    def forward(self,x):
        x = torch.transpose(x,1,3)
        c_max = torch.max(torch.max(torch.max(x,dim=0)[0],dim=0)[0],dim=0)[0].cuda()
        c_min = torch.min(torch.min(torch.min(x,dim=0)[0],dim=0)[0],dim=0)[0].cuda()
                                   
        mean = (c_max+c_min)/2
        var = (c_max-c_min) + self.eps
        # mean = torch.mean(x,(0,1,2)).cuda()
        # var = torch.var(x,(0,1,2)).cuda()
                                   
        if self.training:
            self.moving_mean = self.decay * self.moving_mean + (1-self.decay) * mean
            self.moving_var = self.decay * self.moving_var + (1-self.decay)*var 
            return torch.transpose(self.gamma*(x - mean)/var + self.beta,1,3)
        else:
            return torch.transpose(self.gamma*(x-self.moving_mean)/self.moving_var + self.beta,1,3)



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
