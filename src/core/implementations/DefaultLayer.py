from typing import *
import numpy as np
from src.core.definitions.Neuron import Neuron
from src.core.definitions.Weights import Weights
from src.core.definitions.ActivationFunction import ActivationFunction
from src.core.definitions.FullyConnectedLayer import FullyConnectedLayer
from .FullyConnectedLayerMixin import FullyConnectedLayerMixin
from .DefaultNeuron import DefaultNeuron
from .DefaultWeights import DefaultWeights


class DefaultLayer(FullyConnectedLayer[DefaultNeuron, DefaultWeights], FullyConnectedLayerMixin):

    def __init__(self, neurons: np.ndarray[Neuron], activation_function: ActivationFunction|None, outgoing_weights: Weights | None):
        self.set_neurons_and_outgoing_weights(neurons, outgoing_weights)
        self.__activation_function: ActivationFunction|None = activation_function
        self.__z: np.ndarray[float] = np.zeros_like(self.get_activations())
        self.__dZ: np.ndarray[float] = np.zeros_like(self.__z)

    def forward(self, input: np.ndarray[float]) -> np.ndarray[float]:
        if not input.shape[0] == self.__neuron_count:
            raise ValueError(
                f"Input size ({input.shape[0]}) does not match layer size ({self.__neuron_count})"
            )
        self.__z = input
        self.__dZ = self.__activation_function.derivative(self.__z) if self.__activation_function is not None else None
        activations = self.__activation_function.calculate(self.__z) if self.__activation_function is not None else self.__z
        # set activations in neurons
        for i, neuron in enumerate(self.__neurons):
            neuron.set_activation(activations[i])

        output = activations if self.__outgoing_weights is None else self.__outgoing_weights.as_array() @ activations
        return output

    def get_activations(self) -> np.ndarray[float]:
        return np.array([neuron.get_activation() for neuron in self.__neurons])

    def get_activation_function(self) -> ActivationFunction:
        return self.__activation_function

    def get_weighted_input(self) -> np.ndarray[float]:
        return self.__z

    def get_activation_derivative_at_weighted_inputs(self) -> np.ndarray[float]:
        return self.__dZ

    def get_biases(self) -> np.ndarray[float] | None:
        return np.array([neuron.get_bias() for neuron in self.__neurons])

    def get_outgoing_weights(self) -> Weights | None:
        return self.__outgoing_weights

    def get_neurons(self) -> List[DefaultNeuron]:
        return self.__neurons
    
    def get_neuron_count(self) -> int:
        return len(self.__neurons)

    def set_neurons_and_outgoing_weights(self, neurons: List[DefaultNeuron], outgoing_weights: Weights | None) -> None:
        neurons_count = len(neurons)
        self.__outgoing_weights = None
        
        if outgoing_weights is not None:
            if (not len(outgoing_weights.shape()) == 2):
                raise ValueError(
                    f"Outgoing weights shape ({outgoing_weights.shape()}) is not 2-dimensional"
                )
            if (not outgoing_weights.as_array().dtype == np.float64):
                raise ValueError(
                    f"Outgoing weights dtype ({outgoing_weights.as_array().dtype}) is not float64"
                )
            if (not outgoing_weights.shape()[1] == neurons_count):
                raise ValueError(
                    f"Outgoing weights shape ({outgoing_weights.shape()} does not match layer size ({neurons_count})"
                )
        
        self.__outgoing_weights = outgoing_weights
        self.__neuron_count: int = neurons_count
        self.__neurons: np.ndarray[DefaultNeuron] = neurons

    def add_to_outgoing_weights(self, delta_weights: np.ndarray[float]):
        self.__outgoing_weights.add(delta_weights)

    def multiply_outgoing_weights(self, factor: float):
        self.__outgoing_weights.multiply(factor)

    def shape(self) -> tuple[int, ...]:
        return (self.__neuron_count)
    
    def __len__(self) -> int:
        return self.__neuron_count

    @staticmethod
    def generate(
        neuron_count: int,
        activation_function: ActivationFunction|None = None,
        next_layer_neuron_count: int | None = None,
        weights_initializer: Callable[[tuple[int, int]], DefaultWeights] =
            DefaultWeights.generate,
    ) -> 'DefaultLayer':
        # create default neurons
        neurons = np.array([DefaultNeuron(0.) for _ in range(neuron_count)])
        # create outgoing weights
        outgoing_weights = None if next_layer_neuron_count is None else weights_initializer(
            (next_layer_neuron_count, neuron_count)
        )
        return DefaultLayer(neurons, activation_function, outgoing_weights)