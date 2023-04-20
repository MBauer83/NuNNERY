import unittest
import numpy as np
from src.core.implementations.ReLU import ReLU


class TestReLU(unittest.TestCase):

    def test_relu(self):
        relu = ReLU()
        actual = relu.calculate(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([0.1, 0.2, 0.3, 0., 0., 0.])
        np.testing.assert_allclose(expected, actual)

    def test_relu_derivative(self):
        relu = ReLU()
        actual = relu.derivative(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([1., 1., 1., 0., 0., 0.])
        np.testing.assert_allclose(expected, actual)


if __name__ == '__main__':
    unittest.main()