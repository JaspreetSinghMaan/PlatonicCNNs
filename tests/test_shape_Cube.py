import unittest
import torch


class CubeTests(unittest.TestCase):
  def setUp(self):
    self.Shape = None
    self.args = {}

  def tearDown(self) -> None:
    pass

  def test(self):
    pass
    # self.assertTrue(torch.all(torch.eq(get_round_sum(), 1.)))
    # self.assertTrue(torch.all( > 0.))

  if __name__ == '__main__':
      tests = CubeTests()
      tests.setUp()
      tests.test()