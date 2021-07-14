import unittest
from src.eqCNN import S2SGaugeCNN2D

class EqCNNTests(unittest.TestCase):
  def setUp(self):
    self.batch_size = 1 # random.randint(1,20)
    self.c_in = 1 #random.randint(1,6)
    self.r_in = 1
    self.H = random.randint(3,8)

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
    return
  def test(self, group="C4", model_type='S2S'):
    # input_shape = (batch_size, c_in, r_in, H, W)

    input = torch.rand(size=(self.batch_size, self.c_in, self.r_in, self.H, self.H))

    if model_type == 'S2S' and group == "C4":
      #choose model
      model = S2SGaugeCNN2D(c_in=self.c_in, c_out=4, r_in=self.r_in, r_out=1, kernel_size=3, stride=1, bias=True)

      # choose number of rotations
      num_rotations = 1 #random.randint(0,3)
      output = model(input)
      rot_input = input
      rot_output = output

      for i in range(num_rotations):
        rot_input = group_transform(rot_input)
        rot_output = group_transform(rot_output)

      eq_output = model(rot_input)
      with torch.no_grad():
        if np.allclose(eq_output.numpy(), rot_output.numpy()): # keeping it torch form had issues with gradfn, will fix it later
          print("C4 equivariance test passed!")
        else:
          print("C4 equivariance test failed!")
          print(f"input: {input}")
          print(f"rot_output: {rot_input}")
          print(f"output: {output}")
          print(f"rot_output: {rot_output}")
          print(f"eq_output: {eq_output}")
    
    return 


if __name__ == '__main__':
  tests = EqCNNTests()
  tests.setUp()
  tests.test()