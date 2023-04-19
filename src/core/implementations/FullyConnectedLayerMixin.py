from typing import *
import numpy as np
from core.definitions.Neuron import Neuron
from core.definitions.Weights import Weights
from core.definitions.ActivationFunction import ActivationFunction
from core.definitions.Layer import Layer
from .DefaultNeuron import DefaultNeuron
from .DefaultWeights import DefaultWeights
from .DefaultLayer import DefaultLayer


class FullyConnectedLayerMixin:
    @staticmethod
    def generate(
        size: int,
        next_layer_size: int | None,
        activation_function: ActivationFunction,
        weights_initializer: Callable[[tuple[int, int]], Weights] | None = lambda shape: DefaultWeights.generate(
            shape, lambda s: np.random.nrandom(s)
        ),
        neuron_ctor: Callable[[int], List[Neuron]] = lambda x: [
            DefaultNeuron(0.) for _ in range(x)
        ],
        layer_ctor: Callable[[List[Neuron], ActivationFunction, Weights|None], 'Layer'] =
            lambda nodes, activation, outgoing_weights: DefaultLayer(
                nodes, activation, outgoing_weights
            )
    ) -> 'Layer':
        nodes = neuron_ctor(size)
        return layer_ctor(
            nodes,
            activation_function,
            weights_initializer((size, next_layer_size)) 
            if next_layer_size is not None else None
        )
