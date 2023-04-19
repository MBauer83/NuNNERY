from typing import *
import numpy as np
from .Weights import Weights
from .Neuron import Neuron
from .ActivationFunction import ActivationFunction
from core.implementations import DefaultNeuron, DefaultWeights, DefaultLayer

class Layer:
    def forward(self) -> np.ndarray[float]:
        pass
    def get_activation() -> np.ndarray[float]:
        pass
    def get_weighted_input() -> np.ndarray[float]:
        pass
    def get_biases() -> np.ndarray[float] | None:
        pass
    def get_outgoing_weights() -> Weights | None:
        pass
    def get_nodes() -> List[Neuron]:
        pass
    def set_nodes_and_outgoing_weights(self, nodes: List[Neuron], outgoing_weights: Weights) -> None:
        pass

    @staticmethod
    def generate(
        size: int,
        next_layer_size: int | None,
        activation_function: ActivationFunction, 
        weights_initializer: Callable[[Tuple[int, int]], Weights] | None = lambda shape: DefaultWeights.generate(shape, lambda s: np.random.nrandom(s)),
        nodes_ctor: Callable[[int], List[Neuron]] = lambda x: [DefaultNeuron(0.) for _ in range(x)],
        layer_ctor: Callable[[List[Neuron], ActivationFunction, Weights, np.ndarray[float]|None], 'Layer'] = lambda nodes, activation, outgoing_weights: DefaultLayer(nodes, activation, outgoing_weights)
    ) -> 'Layer':
        nodes = nodes_ctor(size)
        return layer_ctor(
                nodes,
                activation_function,
                weights_initializer((size, next_layer_size)) if next_layer_size is not None else None
            )
