# append base directory `./../../`
import sys
sys.path.append('./../../')

import unittest
import numpy as np
from src.core.implementations.TanH import TanH

class TestTanH(unittest.TestCase):

    def setUp(self):
        self.input = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.tanh = TanH()

    def test_calculate(self):
        expected_output = np.array([0.099668, 0.197375, 0.291313, 0.379949, 0.462117, 0.53705 ])
        np.testing.assert_allclose(self.tanh.calculate(self.input), expected_output, rtol=1e-5)

    def test_derivative(self):
        expected_output = np.array([0.990066, 0.961043, 0.915137, 0.855639, 0.786448, 0.711578])
        np.testing.assert_allclose(self.tanh.derivative(self.input), expected_output, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()

