from typing import *
import numpy as np
from src.core.definitions.Neuron import Neuron
from src.core.definitions.Weights import Weights
from src.core.definitions.ActivationFunction import ActivationFunction
from src.core.definitions.Layer import Layer
from .DefaultNeuron import DefaultNeuron
from .DefaultWeights import DefaultWeights

class FullyConnectedLayerMixin:

    @staticmethod
    def generate(
        size: int,
        next_layer_size: int | None,
        activation_function: ActivationFunction,
        layer_ctor: Callable[[int, ActivationFunction, int|None, Callable[[tuple[int, int]], DefaultWeights]], 'FullyConnectedLayerMixin'],
        neuron_ctor: Callable[[int], list[Neuron]] = lambda x: [
            DefaultNeuron(0.) for _ in range(x)
        ]
    ) -> 'FullyConnectedLayerMixin':
        nodes = neuron_ctor(size)
        weights = DefaultWeights.generate(shape=(size, next_layer_size)) if next_layer_size is not None else None
        return layer_ctor(
            nodes,
            activation_function,
            weights
        )
