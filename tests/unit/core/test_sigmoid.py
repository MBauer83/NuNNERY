# append base directory `./../../`
import sys
sys.path.append('./../../')

import unittest
import numpy as np
from src.core.implementations.Sigmoid import Sigmoid

class TestSigmoid(unittest.TestCase):
    
    def test_sigmoid(self):
        sigmoid = Sigmoid()
        actual = sigmoid.calculate(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([0.52497919, 0.549834, 0.57444252, 0.47502081, 0.450166, 0.42555748])
        np.testing.assert_allclose(expected, actual)

        
    def test_sigmoid_derivative(self):
        sigmoid = Sigmoid()
        actual = sigmoid.derivative(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([0.24937604, 0.24751657, 0.24445831, 0.24937604, 0.24751657, 0.24445831])
        np.testing.assert_allclose(expected, actual)

if __name__ == '__main__':
    unittest.main()
