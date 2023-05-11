from typing import *
import numpy as np
from ..definitions.Neuron import Neuron
from ..definitions.Weights import Weights
from ..definitions.Layer import Layer
from ..definitions.ActivationFunction import ActivationFunction
from ..definitions.FullyConnectedLayer import FullyConnectedLayer, NeuronType_co, WeightType_co
from .FullyConnectedLayerMixin import FullyConnectedLayerMixin
from .DefaultNeuron import DefaultNeuron
from .DefaultWeights import DefaultWeights


class DefaultLayer(FullyConnectedLayer[DefaultNeuron, DefaultWeights], FullyConnectedLayerMixin):

    def __init__(self, neurons: np.ndarray[Neuron], activation_function: ActivationFunction|None, outgoing_weights: Weights | None):
        self._neurons = None
        self._neuron_count = None
        self._outgoing_weights = None
        self.set_neurons_and_outgoing_weights(neurons, outgoing_weights)
        self._activation_function: ActivationFunction | None = activation_function
        self._z: np.ndarray[float] = np.zeros_like(self.get_activations())
        self._dZ: np.ndarray[float] = np.zeros_like(self._z)

    def forward(self, layer_input: np.ndarray[float]) -> np.ndarray[float]:
        if not layer_input.shape[0] == self._neuron_count:
            raise ValueError(
                f"Input size ({layer_input.shape[0]}) does not match layer size ({self._neuron_count})"
            )
        self._z = layer_input
        self._dZ = self._activation_function.derivative(self._z) if self._activation_function is not None else None
        activations = self._activation_function.calculate(self._z) if self._activation_function is not None else self._z
        # set activations in neurons
        for i, neuron in enumerate(self._neurons):
            neuron.set_activation(activations[i])

        output = activations if self._outgoing_weights is None else self._outgoing_weights.as_array() @ activations
        return output

    def get_activations(self) -> np.ndarray[float]:
        return np.array([neuron.get_activation() for neuron in self._neurons])

    def get_activation_function(self) -> ActivationFunction:
        return self._activation_function

    def get_weighted_input(self) -> np.ndarray[float]:
        return self._z

    def get_activation_derivative_at_weighted_inputs(self) -> np.ndarray[float]:
        return self._dZ

    def get_biases(self) -> np.ndarray[float] | None:
        return np.array([neuron.get_bias() for neuron in self._neurons])

    def get_outgoing_weights(self) -> Weights | None:
        return self._outgoing_weights

    def get_neurons(self) -> list[DefaultNeuron]:
        return self._neurons
    
    def get_neuron_count(self) -> int:
        return len(self._neurons)

    def set_neurons_and_outgoing_weights(self, neurons: list[DefaultNeuron], outgoing_weights: Weights | None) -> None:
        neurons_count = len(neurons)
        self._outgoing_weights = None
        
        if outgoing_weights is not None:
            if not len(outgoing_weights.shape()) == 2:
                raise ValueError(
                    f"Outgoing weights shape ({outgoing_weights.shape()}) is not 2-dimensional"
                )
            if not outgoing_weights.as_array().dtype == np.float64:
                raise ValueError(
                    f"Outgoing weights dtype ({outgoing_weights.as_array().dtype}) is not float64"
                )
            if not outgoing_weights.shape()[1] == neurons_count:
                raise ValueError(
                    f"Outgoing weights shape ({outgoing_weights.shape()} does not match layer size ({neurons_count})"
                )
        
        self._outgoing_weights: np.ndarray[float] = outgoing_weights
        self._neuron_count: int = neurons_count
        self._neurons: list[DefaultNeuron] = neurons

    def add_to_outgoing_weights(self, delta_weights: np.ndarray[float]):
        self._outgoing_weights.add(delta_weights)

    def multiply_outgoing_weights(self, factor: float):
        self._outgoing_weights.multiply(factor)

    def shape(self) -> tuple[int, ...]:
        return self._neuron_count
    
    def __len__(self) -> int:
        return self._neuron_count

    @staticmethod
    def generate(
        size: int,
        next_layer_size: int | None,
        activation_function: ActivationFunction,
        weights_initializer: Callable[[tuple[int, int]], Type[WeightType_co]] | None,
        neuron_ctor: Callable[[int], list[Type[NeuronType_co]]],
        layer_ctor: Callable[[list[Type[NeuronType_co]], ActivationFunction, Type[WeightType_co]], Layer]
    ) -> Layer:
        # create default neurons
        neurons = neuron_ctor(size)
        # create outgoing weights
        outgoing_weights = None if next_layer_size is None else weights_initializer(
            (next_layer_size, size)
        )
        return layer_ctor(neurons, activation_function, outgoing_weights)