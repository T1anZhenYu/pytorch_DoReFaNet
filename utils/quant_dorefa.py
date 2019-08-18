import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

class quan_bn(nn.Module):
  def __init__(self,bitA):
    super(quan_bn,self).__init__()
    self.bitA = bitA

  def forward(self,x):

    def get_quan_point():
        return np.array([(2**self.bitA-i+0.5)/(2**self.bitA-1) \
            for i in range(2**self.bitA,1,-1)])

    quan_points0 = get_quan_point().astype(np.float32)

    quan_values = np.array([round((quan_points0[i]-0.005)*(2**self.bitA-1))\
    /(float(2**self.bitA-1)) for i in range(len(quan_points0))])#values after quantization 

    quan_points0 = np.append(np.insert(quan_points0,0,-1000.),np.array([1000.]))

    shape = list(x.size())

    layer = nn.BatchNorm2d(shape[1]).cuda()
    fake_output = layer(x)
    gamma = layer.weight
    beta = layer.bias
    moving_mean = layer.running_mean
    moving_var = torch.sqrt(layer.running_var)

    c_max = torch.max(torch.max(torch.max(x,dim=0).values,dim=-1).values,dim=-1).values
    c_min = torch.min(torch.min(torch.min(x,dim=0).values,dim=-1).values,dim=-1).values

    if self.training:
        bm = torch.unsqueeze((c_max-c_min)/2,dim=-1)
        bv = torch.unsqueeze(torch.sqrt(c_max-c_min),dim=-1)

        quan_points = bv*torch.tensor(quan_points0)/(torch.unsqueeze(gamma,dim=-1)) + \
        bm - bv*toch.unsqueeze(beta/gamma,dim=-1)

    else:
        quan_points = moving_var*quan_points0/(torch.unsqueeze(gamma,dim=-1)) + \
        moving_mean - moving_var*toch.unsqueeze(beta/gamma,dim=-1)

    inputs = torch.reshape(torch.transpose(x,1,-1),[-1,shape[1]])

    label = []

    for i in range(1,len(quan_points)):
        label.append(inputs>quan_points[i-1] * inputs<quan_points[i])

    xn = label[0]*quan_values[0]
    for i in range(1,len(label)):
        xn += label[i]*quan_values[i]

    quan_output = torch.transpose(torch.reshape(xn,[shape[0],shape[2],shape[3],shape[1]]),shape)

    if training:
        return (quan_output - fake_output).detech() + fake_output

    else:
        return quan_output


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
