# append base directory `./../../`
import sys
# get dirname of this file
sys.path.append('./../../')

import unittest
import numpy as np
from unittest.mock import Mock
from src.core.definitions.ActivationFunction import ActivationFunction
from src.core.implementations.DefaultNeuralNetwork import DefaultNeuralNetwork
from src.core.implementations.LeakyReLU import LeakyReLU
from src.core.implementations.Softmax import Softmax
from src.core.implementations.DefaultWeights import DefaultWeights

class TestNeuralNetworkBehavior(unittest.TestCase):
    def setUp(self) -> None:
        class IdentityActivationFn(ActivationFunction):
            @staticmethod
            def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
                return xs
            @staticmethod
            def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
                return np.ones_like(xs)
        # generate a np.array of shape `shape` with elements created by a function n: int -> n*1.5
        seq = lambda n: np.array([(i+1)*1.5 for i in range(n)])

        weights_initializer = lambda shape: DefaultWeights.generate(shape, lambda s: seq(s[0]*s[1]).reshape(s))
        self.network = DefaultNeuralNetwork.generate((2, 2, 2), [IdentityActivationFn, IdentityActivationFn], weights_initializer)
    
    def test_forward(self):
        input = np.array([1, 2])
        output = self.network.forward(input)
        np.testing.assert_array_equal(output, np.array([60.75, 132.75]))

    def test_get_output_activations(self):
        input = np.array([1, 2])
        output = self.network.forward(input)
        np.testing.assert_array_equal(output, self.network.get_output_activations())
    
    def test_shape(self):
        self.assertEqual(self.network.shape(), (2, 2, 2))



if __name__ == '__main__':
    unittest.main()
