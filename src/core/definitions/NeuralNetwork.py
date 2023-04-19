from typing import *
import numpy as np
from .Layer import Layer
from .ActivationFunction import ActivationFunction
from core.implementations import DefaultLayer, DefaultWeights, DefaultNeuralNetwork

class NeuralNetwork:
    def forward(self, input: np.ndarray[float]) -> np.ndarray[float]:
        pass
    def get_output_activations(self) -> np.ndarray[float]:
        pass
    def compile() -> Callable[[np.ndarray[float]], np.ndarray[float]]:
        pass
    @staticmethod
    def generate(
        shape: tuple[int, ...],
        activation_functions: List[ActivationFunction],
    ) -> 'NeuralNetwork':
        # check that activation_functions is of length len(shape) - 1 or len(shape)
        shape_len = len(shape)
        activation_functions_len = len(activation_functions)
        input_has_activation_function = shape_len == activation_functions_len
        if shape_len - 1 != activation_functions_len and not input_has_activation_function:
            raise ValueError("Length of activation_functions must be either len(shape) - 1 or len(shape)")
        layers = []
        if not input_has_activation_function:
            # prepend none to the activation functions
            activation_functions = [None] + activation_functions
        for i in range(shape_len - 1):
            next_layer_shape = shape[i + 1] if i + 1 < shape_len else None
            layers.append(DefaultLayer.generate(shape[i], next_layer_shape, activation_functions[i]))
        return DefaultNeuralNetwork(layers)
