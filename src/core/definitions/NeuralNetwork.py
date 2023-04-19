from abc import ABCMeta, abstractmethod
from typing import Callable, TypeVar, Generic
import numpy as np
from .Layer import Layer

LayerType_co = TypeVar('LayerType_co', covariant=True, bound=Layer)

class NeuralNetwork(Generic[LayerType_co], metaclass=ABCMeta):

    @abstractmethod
    def forward(self, input: np.ndarray[float]) -> np.ndarray[float]:
        raise NotImplementedError
    
    @abstractmethod
    def get_output_activations(self) -> np.ndarray[float]:
        raise NotImplementedError
    
    @abstractmethod
    def compile(self) -> Callable[[np.ndarray[float]], np.ndarray[float]]:
        raise NotImplementedError
    
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError
    
    @abstractmethod
    def get_layer(self, index: int) -> LayerType_co:
        raise NotImplementedError
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, 'forward') and
                callable(subclass.forward) and
                hasattr(subclass, 'get_output_activations') and
                callable(subclass.get_output_activations) and
                hasattr(subclass, 'compile') and
                callable(subclass.compile) and
                hasattr(subclass, 'shape') and
                callable(subclass.shape) and
                hasattr(subclass, 'get_layer') and
                callable(subclass.get_layer)) or
                NotImplemented)