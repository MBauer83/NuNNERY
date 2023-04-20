import unittest
import numpy as np
from src.core.implementations.LeakyReLU import LeakyReLU


class TestLeakyReLU(unittest.TestCase):

    def test_leakyrelu(self):
        leakyrelu = LeakyReLU()
        actual = leakyrelu.calculate(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([0.1, 0.2, 0.3, -0.001, -0.002, -0.003])
        np.testing.assert_allclose(expected, actual)

    def test_leakyrelu_derivative(self):
        leakyrelu = LeakyReLU()
        actual = leakyrelu.derivative(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([1., 1., 1., 0.01, 0.01, 0.01])
        np.testing.assert_allclose(expected, actual)


if __name__ == '__main__':
    unittest.main()