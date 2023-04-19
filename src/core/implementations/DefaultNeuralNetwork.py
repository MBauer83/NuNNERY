from typing import *
import numpy as np
from src.core.definitions.Layer import Layer
from src.core.definitions.FullyConnectedLayer import FullyConnectedLayer
from src.core.definitions.NeuralNetwork import NeuralNetwork
from src.core.definitions.ActivationFunction import ActivationFunction
from .DefaultNeuron import DefaultNeuron
from .DefaultWeights import DefaultWeights
from .DefaultLayer import DefaultLayer


class DefaultNeuralNetwork(NeuralNetwork[DefaultLayer]):
    def __init__(self, layers: List[FullyConnectedLayer[DefaultNeuron, DefaultWeights]]):
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
        self.__layers = layers
        self.__shape = tuple([layer.get_neuron_count() for layer in layers])

    def get_layer(self, index: int) -> Layer:
        if index < 0 or index >= len(self.__layers):
            raise IndexError("Invalid layer index: " + str(index))
        return self.__layers[index]
    
    def shape(self) -> tuple[int, ...]:
        return self.__shape
    
    def get_output_activations(self) -> np.ndarray[float]:
        return self.__layers[-1].get_activations()
    
    def forward(self, input: np.ndarray[float]) -> np.ndarray[float]:
        if not input.shape[0] == self.__shape[0]:
            raise ValueError(
                f"Input size ({input.shape[0]}) does not match layer size ({self.__shape[0]})"
            )
        output = input

        for i, layer in enumerate(self.__layers):
            output = layer.forward(output)
        return output
    
    def compile(self) -> Callable[[np.ndarray[float]], np.ndarray[float]]:
        # Get the vectors and matrices, then return a function which
        # only does the matrix calculations, but don't update anything
        # if f_i is the vectorized activation function of layer i (first layer's is the identity function)
        # and W_i_j is the matrix of weights connecting layer i to layer j, then the expression of a neural network
        # with n layers is: f_n(W_(n-1)_n(f_(n-1)(W_(n-2)_(n-1)(...f_2(W_1_2(Id(x))))...)))
        # where Id(x) is the vectorized identity function applied to x
        
        # get vectorized activation functions for all layers except the first one
        activation_functions = [layer.get_activation_function().calculate for layer in self.__layers[1:]]
        # prepend identity function to the list of vectorized activation functions
        activation_functions.insert(0, lambda x: x)

        # get matrices of weights for all layers except the first one
        matrices_of_weights = [layer.get_outgoing_weights().as_array() for layer in self.__layers[1:]]
        
        def feedforward(w: np.ndarray, f: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
            matmul = np.matmul(w.T, x)
            result = f(matmul)
            return result
        def feed_through(x: np.ndarray) -> np.ndarray:
            zipped = zip(matrices_of_weights, activation_functions)
            val = x
            for weights, activation in zipped:
                # print shape of weights 
                val = feedforward(weights, activation, val)
            return val
        # compose the expression
        expression: Callable[[np.ndarray], np.ndarray] = lambda x: feed_through(x)
        return expression

    @staticmethod
    def generate(
        shape: tuple[int, ...],
        activation_functions: List[ActivationFunction],
        weights_initializer: Callable[[int, int], DefaultWeights] = DefaultWeights.generate
    ) -> 'NeuralNetwork':
        # Check that activation_functions is of length len(shape) - 1 or len(shape)
        shape_len = len(shape)
        activation_functions_len = len(activation_functions)
        input_has_activation_function = shape_len == activation_functions_len
        if shape_len - 1 != activation_functions_len and not input_has_activation_function:
            raise ValueError("Length of activation_functions must be either len(shape) - 1 or len(shape)")
        if not input_has_activation_function:
            # prepend none to the activation functions
            activation_functions = [None] + activation_functions
        layers = []
        for i in range(shape_len):
            # if we are not at the last layer, generate a DefaultWeights instance, otherwise generate a None
            next_layer_shape = shape[i + 1] if i + 1 < shape_len else None
            layer = DefaultLayer.generate(shape[i], activation_functions[i], next_layer_shape, weights_initializer)
            layers.append(layer)
        return DefaultNeuralNetwork(layers)
