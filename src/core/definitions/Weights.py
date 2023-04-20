from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

class Weights(ABC):

    @abstractmethod
    def as_array(self) -> np.ndarray[float]:
        raise NotImplementedError

    @abstractmethod
    def add(self, weights_delta: np.ndarray[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def multiply(self, factor: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def generate(shape: tuple[int, int], initializer: Callable[[int], np.ndarray[float]]) -> 'Weights':
        raise NotImplementedError
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, 'as_array') and
                callable(subclass.as_array) and
                hasattr(subclass, 'add') and
                callable(subclass.add) and
                hasattr(subclass, 'multiply') and
                callable(subclass.multiply) and
                hasattr(subclass, 'shape') and
                callable(subclass.shape) and
                hasattr(subclass, '__len__') and
                callable(subclass.__len__) and
                hasattr(subclass, 'generate') and
                callable(subclass.generate)) or
                NotImplemented)

    