import unittest
import numpy as np
from src.core.implementations.SeLU import SeLU

class TestSeLU(unittest.TestCase):
    
    def test_selu(self):
        selu = SeLU()
        actual = selu.calculate(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([0.1050702, 0.21014041, 0.31521061, -0.16730467, -0.31868911, -0.45566743])
        np.testing.assert_allclose(expected, actual, rtol=1e-04, atol=0)
        
    def test_selu_derivative(self):
        selu = SeLU()
        actual = selu.derivative(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([1.050701, 1.050701, 1.050701, 1.590794, 1.43941 , 1.302432])
        np.testing.assert_allclose(expected, actual, rtol=1e-04, atol=0)

if __name__ == '__main__':
    unittest.main()
