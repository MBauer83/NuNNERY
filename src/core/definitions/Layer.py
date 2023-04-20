from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):

    @abstractmethod
    def forward(self, input: np.ndarray[float]) -> np.ndarray[float]:
        raise NotImplementedError

    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, 'forward') and
                callable(subclass.forward) and
                hasattr(subclass, 'shape') and
                callable(subclass.shape) and
                hasattr(subclass, '__len__') and
                callable(subclass.__len__)) or
                NotImplemented)