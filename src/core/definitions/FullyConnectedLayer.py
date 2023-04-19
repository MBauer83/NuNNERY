from abc import ABCMeta, abstractmethod
from typing import List, Callable, Generic, Type, TypeVar
import numpy as np
from .Neuron import Neuron
from .Weights import Weights
from .ActivationFunction import ActivationFunction
from .Layer import Layer

NeuronType_co = TypeVar('NeuronType_co', covariant=True, bound=Neuron)
WeightType_co = TypeVar('WeightType_co', covariant=True, bound=Weights)

class FullyConnectedLayer(Layer, Generic[NeuronType_co, WeightType_co]):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_activations(self) -> np.ndarray[float]:
        raise NotImplementedError
    
    @abstractmethod
    def get_weighted_input(self) -> np.ndarray[float]:
        raise NotImplementedError
    
    @abstractmethod
    def get_biases(self) -> np.ndarray[float] | None:
        raise NotImplementedError
    
    @abstractmethod
    def get_outgoing_weights(self) -> WeightType_co | None:
        raise NotImplementedError
    
    @abstractmethod
    def get_neurons(self) -> List[NeuronType_co]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate(
        size: int,
        next_layer_size: int | None,
        activation_function: ActivationFunction, 
        weights_initializer: Callable[[tuple[int, int]], Type[WeightType_co]] | None,
        neuron_ctor: Callable[[int], List[Type[NeuronType_co]]],
        layer_ctor: Callable[[List[Type[NeuronType_co]], ActivationFunction, Type[WeightType_co]], 'Layer']
    ) -> 'Layer':
        raise NotImplementedError
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, 'get_activations') and
                callable(subclass.get_activations) and
                hasattr(subclass, 'get_weighted_input') and
                callable(subclass.get_weighted_input) and
                hasattr(subclass, 'get_biases') and
                callable(subclass.get_biases) and
                hasattr(subclass, 'get_outgoing_weights') and
                callable(subclass.get_outgoing_weights) and
                hasattr(subclass, 'get_neurons') and
                callable(subclass.get_neurons) and
                hasattr(subclass, 'get_weights') and
                callable(subclass.get_weights) and
                hasattr(subclass, 'generate') and
                callable(subclass.generate)) or
                NotImplemented)
