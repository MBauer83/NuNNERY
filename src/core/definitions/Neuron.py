from abc import ABC, abstractmethod

class Neuron(ABC):

    @abstractmethod
    def get_activation(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_bias(self) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def set_activation(self, activation: float) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def set_bias(self, bias: float) -> None:
        raise NotImplementedError
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, 'get_activation') and
                callable(subclass.get_activation) and
                hasattr(subclass, 'get_bias') and
                callable(subclass.get_bias) and
                hasattr(subclass, 'set_activation') and
                callable(subclass.set_activation) and
                hasattr(subclass, 'set_bias') and
                callable(subclass.set_bias)) or
                NotImplemented)
