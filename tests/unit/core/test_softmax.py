# append base directory `./../../`
import sys
sys.path.append('./../../')

import unittest
import numpy as np
from src.core.implementations.Softmax import Softmax

class TestSoftmax(unittest.TestCase):

    def setUp(self):
        self.input = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.softmax = Softmax()

    def test_calculate(self):
        expected_output = np.array([0.127927, 0.141381, 0.15625 , 0.172683, 0.190844, 0.210915])
        np.testing.assert_allclose(self.softmax.calculate(self.input), expected_output, rtol=1e-5)

    def test_derivative(self):
        expected_output = np.array([0.111561, 0.121392, 0.131836, 0.142864, 0.154423, 0.16643 ])
        np.testing.assert_allclose(self.softmax.derivative(self.input), expected_output, rtol=1e-5)

if __name__ == '__main__':
    unittest.main()
