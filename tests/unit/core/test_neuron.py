# append base directory `./../../`
import sys
sys.path.append('./../../')

import unittest
from src.core.implementations.DefaultNeuron import DefaultNeuron

class TestDefaultNeuron(unittest.TestCase):

    def setUp(self):
        self.activation = 0.5
        self.bias = 0.1
        self.neuron = DefaultNeuron(self.activation, self.bias)

    def test_get_activation(self):
        self.assertEqual(self.neuron.get_activation(), self.activation)

    def test_get_bias(self):
        self.assertEqual(self.neuron.get_bias(), self.bias)

    def test_set_activation(self):
        new_activation = 0.3
        self.neuron.set_activation(new_activation)
        self.assertEqual(self.neuron.get_activation(), new_activation)

    def test_set_bias(self):
        new_bias = 0.2
        self.neuron.set_bias(new_bias)
        self.assertEqual(self.neuron.get_bias(), new_bias)

if __name__ == '__main__':
    unittest.main()
