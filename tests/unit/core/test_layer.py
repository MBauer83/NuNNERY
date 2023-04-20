# append base directory `./../../`
import sys
sys.path.append('./../../')


import unittest
import numpy as np
from functools import reduce
from src.core.implementations.DefaultWeights import DefaultWeights
from src.core.implementations.DefaultLayer import DefaultLayer
from src.core.implementations.DefaultNeuron import DefaultNeuron
from src.core.implementations.Identity import Identity

from src.core.implementations.Identity import Identity
from src.core.implementations.DefaultNeuron import DefaultNeuron
from src.core.implementations.DefaultWeights import DefaultWeights
from src.core.implementations.DefaultLayer import DefaultLayer


class TestDefaultLayer(unittest.TestCase):

    def setUp(self):
        self.output_size = 3
        self.input_size = 2
        self.input_data = np.array([1., 2.])
        self.neurons = [DefaultNeuron(0.) for _ in range(self.input_size)]
        mul_lambda = lambda x, y: x*y
        val_gen = lambda n: (n+1)*0.5
        weights_gen = lambda s: np.array([val_gen(i) for i in range(reduce(mul_lambda, s))]).reshape(s)
        self.outgoing_weights = DefaultWeights.generate(
            (self.output_size, self.input_size),
            weights_gen
        )
        self.activation_function = Identity()
        self.layer = DefaultLayer(
            self.neurons,
            self.activation_function,
            self.outgoing_weights
        )

    def test_forward(self):
        output = self.layer.forward(self.input_data)
        expected_output = np.array([2.5, 5.5, 8.5])
        np.testing.assert_equal(expected_output, output)

    def test_forward_with_wrong_input_size_raises_error(self):
        input_data = np.array([1.])
        with self.assertRaises(ValueError):
            self.layer.forward(input_data)

    def test_get_activations(self):
        activations = self.layer.get_activations()
        self.assertEqual(activations.shape[0], self.input_size)

    def test_get_activation_function(self):
        activation_function = self.layer.get_activation_function()
        self.assertIsInstance(activation_function, Identity)

    def test_get_weighted_input(self):
        weighted_input = self.layer.get_weighted_input()
        self.assertEqual(weighted_input.shape[0], self.input_size)

    def test_get_activation_derivative_at_weighted_inputs(self):
        dZ = self.layer.get_activation_derivative_at_weighted_inputs()
        self.assertEqual(dZ.shape[0], self.input_size)

    def test_get_biases(self):
        biases = self.layer.get_biases()
        np.testing.assert_equal(biases, np.zeros(self.input_size))

    def test_get_outgoing_weights(self):
        outgoing_weights = self.layer.get_outgoing_weights()
        self.assertIsNotNone(outgoing_weights)
        self.assertEqual(outgoing_weights.as_array().shape, (self.output_size, self.input_size))

    def test_get_neurons(self):
        neurons = self.layer.get_neurons()
        self.assertEqual(len(neurons), self.input_size)

    def test_get_neuron_count(self):
        neuron_count = self.layer.get_neuron_count()
        self.assertEqual(neuron_count, self.input_size)

    def test_set_neurons_and_outgoing_weights(self):
        new_neurons = [DefaultNeuron(0.) for _ in range(self.input_size + 1)]
        new_outgoing_weights = DefaultWeights.generate(
            (self.input_size + 1, self.output_size))
        self.layer.set_neurons_and_outgoing_weights(
            new_neurons,
            new_outgoing_weights
        )
        self.assertEqual(len(self.layer.get_neurons()), self.input_size + 1)
        self.assertIsNotNone(self.layer.get_outgoing_weights())
        self.assertEqual(self.layer.get_outgoing_weights().as_array().shape, (self.output_size, self.input_size + 1))

    def test_set_neurons_and_outgoing_weights_with_wrong_outgoing_weights_shape_raises_error(self):
        new_neurons = [DefaultNeuron(0.) for _ in range(self.input_size + 1)]
        new_outgoing_weights = DefaultWeights.generate(
            (self.input_size + 1, self.output_size + 1))
        with self.assertRaises(ValueError):
            self.layer.set_neurons_and_outgoing_weights(
                new_neurons,
                new_outgoing_weights
            )
       
