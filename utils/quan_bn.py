import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#N,C,W,H
def quan_bn(x,training,bitW,bitA,kernel_size):
    def get_quan_point():
        return np.array([(2**bitA-i+0.5)/(2**bitA-1) \
            for i in range(2**bitA,1,-1)])

    quan_points0 = get_quan_point().astype(np.float32)

    quan_values = np.array([round((quan_points0[i]-0.005)*(2**bitA-1))\
    /(float(2**bitA-1)) for i in range(len(quan_points0))])#values after quantization 

    shape = list(x.size())

    def dignoal(x,kernel_size):
        dig = torch.eye(kernel_size)
        dig = dig.repeat(math.ceil(shape[2]/kernel_size),\
            math.ceil(shape[3]/kernel_size))[:shape[2],:shape[3]]
        dig = torch.unsqueeze(torch.unsqueeze(dig,0),0).repeat(shape[0],shape[1],1,1)

        x_ = x*dig

        num = shape[0]*(math.floor(shape[2]/kernel_size)*math.floor(shape[3]/kernel_size)*kernel_size+\
        shape[2]%kernel_size*math.floor(shape[3]/kernel_size) +\
        shape[3]%kernel_size*math.floor(shape[2]/kernel_size) +\
        min(shape[2]%kernel_size,shape[3]%kernel_size))

        ave = torch.sum(x_,dim=[0,2,3])/num
        return ave

    layer = nn.BatchNorm2d(shape[1])
    fake_output = layer(x)
    gamma = layer.weight
    beta = layer.bias
    moving_mean = layer.running_mean
    moving_var = layer.running_var

    if training:
        bm = torch.unsqueeze(dignoal(x,kernel_size),dim=-1)
        bv = torch.unsqueeze(moving_var,dim=-1)

        quan_points = bv*quan_points0/(torch.unsqueeze(gamma,dim=-1)) + \
        bm - bv*toch.unsqueeze(beta/gamma,dim=-1)
    else:

        quan_points = moving_mean*quan_points0/(torch.unsqueeze(gamma,dim=-1)) + \
        moving_mean - moving_mean*toch.unsqueeze(beta/gamma,dim=-1)

    inputs = torch.transpose(x,1,-1)


    output = torch.zeros(inputs.size())
    for i in range(len(quan_points)):
        if i<len(quan_points)-1:
            label1 = inputs<quan_points[:,i+1]
            label2 = inputs>quan_points[:,i]

            output += (label2*label1).float()*quan_values[i+1]
        else:
            label = inputs>quan_points[:,i]

            output += label.float()
    output = torch.transpose(output,1,-1)
    return (output - fake_output).detach() +fake_output







