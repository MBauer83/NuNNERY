from typing import *
import numpy as np
from core.definitions.Layer import Layer
from core.definitions.FullyConnectedLayer import FullyConnectedLayer
from core.definitions.NeuralNetwork import NeuralNetwork
from core.definitions.ActivationFunction import ActivationFunction
from .NeuralNetworkMixin import NeuralNetworkMixin


class DefaultNeuralNetwork(NeuralNetwork, NeuralNetworkMixin):
    def __init__(self, layers: List[FullyConnectedLayer]):
        # check that there are at least 2 layers
        if len(layers) < 2:
            raise ValueError("Invalid neural network initialization - at least 2 layers are required. Actual number of layers: " + str(len(layers)) + ".")
        # iterate over layers 1..n-1 and check that the first dimension of the outgoing weights of each layer
        # is equal to the flattened shape of the activations of the next layer
        for i in range(len(layers) - 1):
            current_layer: Layer = layers[i]
            next_layer: Layer = layers[i + 1]
            current_layer_activations = current_layer.get_activations()
            next_layer_activations = next_layer.get_activations()
            current_layer_activations_flattened_shape = np.prod(current_layer_activations.shape)
            next_layer_activations_flattened_shape = np.prod(next_layer_activations.shape)
            if current_layer_activations_flattened_shape != next_layer_activations_flattened_shape:
                raise ValueError("Invalid neural network initialization - the flattened shape of the activations of layer " + str(i) + " is not equal to the flattened shape of the activations of layer " + str(i + 1) + ".")
        self.layers = layers
        self.output_layer_dimension = self.layers[-1].get_node_count()
        print(f'Neural network initialized with {len(self.layers)} layers.')


    @staticmethod
    def generate(
        shape: tuple[int, ...],
        activation_functions: List[ActivationFunction],
    ) -> 'NeuralNetwork':
        pass