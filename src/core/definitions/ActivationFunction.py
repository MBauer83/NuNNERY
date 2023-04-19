from abc import ABCMeta, abstractmethod
import numpy as np

class ActivationFunction(metaclass=ABCMeta):

    @staticmethod
    @abstractmethod
    def calculate(xs: np.ndarray[float]) -> np.ndarray[float]:
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def derivative(xs: np.ndarray[float]) -> np.ndarray[float]:
        raise NotImplementedError
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'calculate') and
                callable(subclass.calculate) and
                hasattr(subclass, 'derivative') and
                callable(subclass.derivative) or
                NotImplemented)