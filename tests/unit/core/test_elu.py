# append base directory `./../../`
import sys
sys.path.append('./../../')

import unittest
import numpy as np
from src.core.implementations.ELU import ELU

class TestELU(unittest.TestCase):
    
    def test_elu_is_correct(self):
        elu = ELU()
        actual = elu.calculate(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([ 0.1     ,  0.2     ,  0.3     , -0.095163, -0.181269, -0.259182])
        np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=0)

        
    def test_elu_derivative_is_correct(self):
        elu = ELU()
        actual = elu.derivative(np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))
        expected = np.array([1, 1, 1, 0.90483742, 0.81873075, 0.74081822])
        np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=0)

if __name__ == '__main__':
    unittest.main()