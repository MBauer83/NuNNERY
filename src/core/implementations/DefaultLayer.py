from typing import *
import numpy as np
from core.definitions.Neuron import Neuron
from core.definitions.Weights import Weights
from core.definitions.ActivationFunction import ActivationFunction
from core.definitions.FullyConnectedLayer import FullyConnectedLayer
from .FullyConnectedLayerMixin import FullyConnectedLayerMixin
from .DefaultNeuron import DefaultNeuron


class DefaultLayer(FullyConnectedLayer[DefaultNeuron], FullyConnectedLayerMixin):
    _neurons: np.ndarray[Neuron]
    _neuron_count: int
    _outgoing_weights: Weights | None
    _activation_function: ActivationFunction
    _z: np.ndarray[float]
    _dZ: np.ndarray[float]

    def __init__(self, neurons: np.ndarray[Neuron], activation_function: ActivationFunction, outgoing_weights: Weights | None):
        self.set_neurons_and_outgoing_weights(neurons, outgoing_weights)
        self.__activation_function: ActivationFunction = activation_function
        self.__z: np.ndarray[float] = np.zeros_like(self.get_activations())
        self.__dZ: np.ndarray[float] = np.zeros_like(self.get_z())

    def forward(self, input: np.ndarray[float]) -> np.ndarray[float]:
        if not input.shape[0] == self.__neuron_count:
            raise ValueError(
                f"Input size ({input.shape[0]}) does not match layer size ({self.__neuron_count})"
            )
        self.__z = input
        self.__dZ = self.__activation_function.derivative(self.__z)
        activations = self.__activation_function(self.__z)
        # set activations in neurons
        for i, neuron in enumerate(self.__neurons):
            neuron.set_activation(activations[i])
        output = activations if self.__outgoing_weights is None else self.__outgoing_weights.as_array() @ activations
        return output

    def get_activations(self) -> np.ndarray[float]:
        return np.array([neuron.get_activations() for neuron in self.__neurons])

    def get_weighted_input(self) -> np.ndarray[float]:
        return self.__z

    def get_biases(self) -> np.ndarray[float] | None:
        return np.array([neuron.get_bias() for neuron in self.__neurons])

    def get_outgoing_weights(self) -> Weights | None:
        return self.__outgoing_weights

    def get_neurons(self) -> List[DefaultNeuron]:
        return self.__neurons

    def set_neurons_and_outgoing_weights(self, neurons: List[DefaultNeuron], outgoing_weights: Weights|None) -> None:
        neurons_shape = neurons.shape
        neurons_count = np.prod(neurons.shape)
        self.__outgoing_weights = None
        if outgoing_weights is not None and (
            (not len(outgoing_weights.shape()) == 2) or
            (not outgoing_weights.weights.dtype == np.float64) or
            (not outgoing_weights.weights.shape[1] == neurons_count)
        ):
            raise ValueError(
                f"Outgoing weights shape ({outgoing_weights.weights.shape}) does not match layer size ({neurons_count})"
            )
        self.__outgoing_weights = outgoing_weights
            
        self.__neurons_shape: tuple = neurons_shape
        self.__neuron_count: int = neurons_count
        self.__neurons: np.ndarray[DefaultNeuron] = neurons
    
    def add_to_outgoing_weights(self, delta_weights: np.ndarray[float]):
        self.__outgoing_weights.add(delta_weights)

    def multiply_outgoing_weights(self, factor: float):
        self.__outgoing_weights.multiply(factor)
