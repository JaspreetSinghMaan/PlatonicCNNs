import unittest
import random
import numpy as np
from src.eqCNN import CubeConv2D

class EqCNNTests(unittest.TestCase):
  def setUp(self):
    self.batch_size = random.randint(1,20)
    self.c_out = 2
    self.r_out = 1
    self.c_in = random.randint(1,6)
    self.r_in = 1
    self.H = random.randint(3,8)
    #print(f"batch_size: {self.batch_size}")
    #print(f"c_in: {self.c_in}")
    #print(f"r_in: {self.r_in}")
    #print(f"H: {self.H}")

  def group_transform(self, input, group='C4'):
    input_shape = input.size()
    assert input_shape[-1] == input_shape[-2], "The input must be a square"

    if group == 'C4':
      # rotates input by 90 degrees clockwise
      input_c4 = torch.rand(input_shape)
      for i in range(input_shape[-2]):
        for j in range(input_shape[-1]):
          input_c4[:,:,:,i,j] = input[:,:,:,input_shape[-1]-1-j,i]
      return input_c4
    
    if group == 'Zn':
      # translates input by 1 unit
      input_zn = torch.rand(input_shape) # shape: (batch_size, c_out, r_out, H, H)
      for i in range(input_shape[-3]):
        input_zn[:,:,i,:,:] = input[:,:,(i+1)%4,:,:]
      return input_zn

    return


  def test(self, group="C4", symmetry_type='S2S'):
    # input_shape = (batch_size, c_in, r_in, H, W)
    if symmetry_type == 'S2R':
      self.r_out = 4
    elif symmetry_type == 'R2R':
      self.r_in = 4
      self.r_out = 4
    input = torch.rand(size=(self.batch_size, self.c_in, self.r_in, self.H, self.H))

    #choose model
    model = CubeConv2D(c_in=self.c_in, c_out=self.c_out, r_in=self.r_in, r_out=self.r_out, kernel_size=3, stride=1, symmetry_type=symmetry_type, bias=True)

    # choose number of rotations
    num_rotations = 1 # random.randint(0,3); changing the number of rotations needs the number of translations to be changed, will do that later
    output = model(input)
    rot_input = input
    rot_output = output

    for i in range(num_rotations):
      rot_input = self.group_transform(rot_input, group=group)
      rot_output = self.group_transform(rot_output, group=group)

    eq_output = model(rot_input)
    # print(f"eq_output: {eq_output.size()}")
    with torch.no_grad():
      if symmetry_type == 'S2S':
        if np.allclose(eq_output.numpy(), rot_output.numpy(), rtol=1e-03, atol=1e-03, equal_nan=False): # keeping it torch form had issues with gradfn, will fix it later
          print("S2S C4 equivariance test passed!")
        else:
          print("S2S C4 equivariance test failed!")
          print(f"input: {torch.tensor(input.tolist())}")
          print(f"rot_input: {torch.tensor(rot_input.tolist())}")
          print(f"output: {torch.tensor(output.tolist())}")
          print(f"rot_output: {torch.tensor(rot_output.tolist())}")
          print(f"eq_output: {torch.tensor(eq_output.tolist())}")
      elif symmetry_type == 'S2R' or symmetry_type == 'R2R':
        trans_eq_output = self.group_transform(eq_output, group='Zn')
        if np.allclose(trans_eq_output.numpy(), rot_output.numpy(), rtol=1e-03, atol=1e-03, equal_nan=False): # keeping it torch form had issues with gradfn, will fix it later
          print(symmetry_type + " C4 equivariance test passed!")
        else:
          print(symmetry_type + " C4 equivariance test failed!")
          print(f"input: {torch.tensor(input.tolist())}")
          print(f"rot_input: {torch.tensor(rot_input.tolist())}")
          print(f"output: {torch.tensor(output.tolist())}")
          print(f"rot_output: {torch.tensor(rot_output.tolist())}")
          print(f"eq_output: {torch.tensor(eq_output.tolist())}")
      return 


if __name__ == '__main__':
  for i in range(100):
    print(f"Test no.: {i}")
    tests = EqCNNTests()
    tests.setUp()
    tests.test(symmetry_type='S2S')
    tests.test(symmetry_type='S2R')
    tests.test(symmetry_type='R2R')