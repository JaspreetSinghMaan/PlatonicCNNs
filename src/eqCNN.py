import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math

# Scalar to scalar (isotropic) 2D G-CNN
class S2SCubeConv2D(nn.Module):
  
  def __init__(self, c_in, c_out, r_in, r_out, kernel_size=3, stride=1, bias=True):
    '''
    r_in = r_out = 1 for S2S
    Input dimensions: (batch_size; c_in; r_in; H; H )
    No padding here, we assume the input is padded
    Output dimensions: (batch_size; c_out; r_out; H'; H' ) ; H' needs to be computed based on stride and kernel_size
    '''
    super(S2SCubeConv2D, self).__init__()

    self.c_in = c_in
    self.c_out = c_out
    self.r_in = r_in
    self.r_out = r_out
    self.kernel_size = kernel_size
    self.stride = stride
    self.bias = bias
    self.padding = 0

    self.weight = Parameter(torch.Tensor(self.c_out, self.c_in, self.r_in, 3)) # because there are 3 independent parameters per filter
                            
    if self.bias:
      self.bias = Parameter(torch.Tensor(c_out))
    else:
      self.bias = self.register_parameter('bias', None)

    self.reset_parameters()    
    self.eq_indices = self.get_eq_indices()  # currently implemented for S2S, but modify it for S2R and R2R as and when needed

  def reset_parameters(self):
    n = self.c_in * self.kernel_size ** 2
    stdv = 1./math.sqrt(n)

    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
      self.bias.data.uniform_(-stdv, stdv)
  
  def get_eq_indices(self, symmetry_type='S2S'):
    eq_indices = None
    if symmetry_type == 'S2S':
      eq_indices = torch.tensor([0, 1, 0, 1, 2, 1, 0, 1, 0]).view(3,3) # eq_indices for the S2S case
    return eq_indices
  
  def forward(self, input):
    eq_weight_shape = (self.c_out, self.c_in, self.kernel_size, self.kernel_size)
    eq_weight = self.weight
    if self.eq_indices is not None:
      eq_weight = self.weight[:,:,:, self.eq_indices]
    eq_weight = eq_weight.view(eq_weight_shape)

    input_shape = input.size()
    input = input.view(input_shape[0], self.c_in*self.r_in, input_shape[-2], input_shape[-1]) # input_shape[0] = batch_size, (input_shape[-2], input_shape[-1]) = (H, W)
    output = F.conv2d(input, weight=eq_weight, bias=None, stride=self.stride, padding=self.padding)
    batch_size, _, nx_out, ny_out = output.size() # _ = channel_out = c_out*r_out; we will convert it to c_out*r_out
    output = output.view(batch_size, self.c_out, self.r_out, ny_out, nx_out)

    if self.bias is not None:
      bias = self.bias.view(1, self.c_out, 1, 1, 1)
      output = output + bias
    
    return output