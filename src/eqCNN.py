import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch
import math

class CubeConv2D(nn.Module):
  
  def __init__(self, c_in, c_out, r_in, r_out=4, kernel_size=3, stride=1, symmetry_type = 'S2R', bias=True):
    '''
    symmetry_type = 'S2S' or 'S2R' or 'R2R'
    for S2S: r_in = r_out = 1 
    for S2R: r_in = 1, r_out = 4
    for R2R: r_in = 4, r_out = 4
    Input dimensions: (batch_size; c_in; r_in; H; H )
    
    No padding here, we assume the input is padded
    Output dimensions: (batch_size; c_out; r_out; H'; H' ) ; H' needs to be computed based on stride and kernel_size

    filter dimensions: (c_out, r_out, c_in, r_in, kernel_size, kernel_size)
    '''
    super(CubeConv2D, self).__init__()

    self.c_in = c_in
    self.c_out = c_out
    self.r_in = r_in
    self.r_out = r_out
    self.kernel_size = kernel_size
    self.stride = stride
    self.bias = bias
    self.padding = 0
    self.symmetry_type = symmetry_type

    self.weight = Parameter(torch.Tensor(self.c_out, self.c_in, self.r_in, self.kernel_size)) # because there are 3 independent parameters per filter
    
    if self.symmetry_type == 'S2R' or self.symmetry_type == 'R2R':
      self.weight = Parameter(torch.Tensor(self.c_out, self.c_in, self.r_in, self.kernel_size, self.kernel_size))
             
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

  def rotate_filter(self, input):
    input = torch.tensor(input)
    input_shape = input.size()
    assert input_shape[-1] == input_shape[-2], "The filter must be of square shape"
    # rotates input by 90 degrees clockwise
    input_c4 = torch.rand(input_shape)
    for i in range(input_shape[-2]):
      for j in range(input_shape[-1]):
        input_c4[i,j] = input[input_shape[-1]-1-j,i]
    return input_c4.long()
  
  def get_eq_indices(self):
    eq_indices = None
    if self.symmetry_type == 'S2S':
      eq_indices = torch.tensor([0, 1, 0, 1, 2, 1, 0, 1, 0]).view(3,3) # eq_indices for the S2S case
    elif self.symmetry_type == 'S2R' or self.symmetry_type == 'R2R':
      eq_indices = []
      eq_indices += [torch.tensor([i for i in range(9)]).view(3,3).tolist()]
      eq_indices += [self.rotate_filter(eq_indices[-1]).tolist()]
      eq_indices += [self.rotate_filter(eq_indices[-1]).tolist()]
      eq_indices += [self.rotate_filter(eq_indices[-1]).tolist()]
      eq_indices = torch.tensor(eq_indices)
    return eq_indices
  
  def forward(self, input):
    eq_weight_shape = (self.c_out*self.r_out, self.c_in*self.r_in, self.kernel_size, self.kernel_size)
    eq_weight = self.weight

    if self.eq_indices is not None:

      if self.symmetry_type == 'S2S':
        eq_weight = eq_weight[:,:,:, self.eq_indices]

      elif self.symmetry_type == 'S2R' or self.symmetry_type == 'R2R':
        eq_weight = eq_weight.view(self.c_out, self.c_in, self.r_in, self.kernel_size ** 2) # shape = (c_out, c_in, r_in, kernel_size*kernel_size)
        eq_weight = eq_weight[:,:,:, self.eq_indices] # shape = (c_out, c_in, r_in, r_out, kernel_size, kernel_size)
        eq_weight = eq_weight.permute(0, 3, 1, 2, 4, 5) # shape = (c_out, r_out, c_in, r_in, kernel_size, kernel_size)

    eq_weight = eq_weight.reshape(eq_weight_shape)

    input_shape = input.size()
    input = input.view(input_shape[0], self.c_in*self.r_in, input_shape[-2], input_shape[-1]) # input_shape[0] = batch_size, (input_shape[-2], input_shape[-1]) = (H, W)
    
    output = F.conv2d(input, weight=eq_weight, bias=None, stride=self.stride, padding=self.padding)
    batch_size, _, nx_out, ny_out = output.size() # _ = channel_out = c_out*r_out; we will convert it to (c_out, r_out)
    output = output.view(batch_size, self.c_out, self.r_out, ny_out, nx_out)

    if self.bias is not None:
      bias = self.bias.view(1, self.c_out, 1, 1, 1)
      output = output + bias
    
    return output