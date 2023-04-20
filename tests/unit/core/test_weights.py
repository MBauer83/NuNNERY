# append base directory `./../../`
import sys
sys.path.append('./../../')

import unittest
import numpy as np
from src.core.implementations.DefaultWeights import DefaultWeights

class TestDefaultWeights(unittest.TestCase):
    
    def setUp(self):
        self.weights_data = np.array([[1., 2.], [3., 4.]])
        self.weights = DefaultWeights(self.weights_data)
    
    def test_get_weights(self):
        np.testing.assert_array_equal(self.weights.get_weights(), self.weights_data)

    def test_set_weights(self):
        new_weights_data = np.array([[5., 6.], [7., 8.]])
        self.weights.set_weights(new_weights_data)
        np.testing.assert_array_equal(self.weights.get_weights(), new_weights_data)

    def test_add(self):
        delta_weights = np.array([[1., 1.], [1., 1.]])
        expected_result = self.weights_data + delta_weights
        self.weights.add(delta_weights)
        np.testing.assert_array_equal(self.weights.get_weights(), expected_result)

    def test_multiply(self):
        factor = 2.
        expected_result = self.weights_data * factor
        self.weights.multiply(factor)
        np.testing.assert_array_equal(self.weights.get_weights(), expected_result)
    
    def test_as_array(self):
        np.testing.assert_array_equal(self.weights.as_array(), self.weights_data)

    def test_shape(self):
        expected_shape = self.weights_data.shape
        self.assertEqual(self.weights.shape(), expected_shape)
    
    def test_len(self):
        expected_length = self.weights_data.shape[0] * self.weights_data.shape[1]
        self.assertEqual(len(self.weights), expected_length)

    def test_generate(self):
        shape = (3, 4)
        weights = DefaultWeights.generate(shape)
        self.assertEqual(weights.shape(), shape)

if __name__ == '__main__':
    unittest.main()
