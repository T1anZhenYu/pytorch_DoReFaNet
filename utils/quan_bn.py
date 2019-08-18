import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
#N,C,W,H
def quan_bn(x,training,bitA,kernel_size):
    def get_quan_point():
        return np.array([(2**bitA-i+0.5)/(2**bitA-1) \
            for i in range(2**bitA,1,-1)])

    quan_points0 = get_quan_point().astype(np.float32)

    quan_values = np.array([round((quan_points0[i]-0.005)*(2**bitA-1))\
    /(float(2**bitA-1)) for i in range(len(quan_points0))])#values after quantization 

    quan_points0 = np.append(np.insert(quan_points0,0,-1000.),np.array([1000.]))

    shape = list(x.size())


    layer = nn.BatchNorm2d(shape[1])
    fake_output = layer(x)
    gamma = layer.weight
    beta = layer.bias
    moving_mean = layer.running_mean
    moving_var = torch.sqrt(layer.running_var)

    c_max = torch.max(torch.max(torch.max(x,dim=0).values,dim=-1).values,dim=-1)
    c_min = torch.min(torch.min(torch.min(x,dim=0).values,dim=-1).values,dim=-1)
    if training:
        bm = torch.unsqueeze((c_max-c_min)/2,dim=-1)
        bv = torch.unsqueeze(torch.sqrt(c_max-c_min),dim=-1)

        quan_points = bv*quan_points0/(torch.unsqueeze(gamma,dim=-1)) + \
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

    return 







