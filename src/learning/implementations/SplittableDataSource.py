from abc import ABC, abstractmethod
import numpy as np
from .DataSource import DataSource

class SplittableDataSource(DataSource, ABC):
    @abstractmethod
    def split(self, splits: tuple[float, ...]) -> tuple[DataSource, ...]:
        raise NotImplementedError()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return ((hasattr(subclass, '__len__') and
                callable(subclass.__len__) and
                hasattr(subclass, 'shape') and
                callable(subclass.shape) and
                hasattr(subclass, 'get_next_batch') and
                callable(subclass.get_next_batch) and
                hasattr(subclass, 'get_all_data') and
                callable(subclass.get_all_data) and
                hasattr(subclass, 'reset') and
                callable(subclass.reset) and
                hasattr(subclass, 'split') and
                callable(subclass.split)) or
                NotImplemented)