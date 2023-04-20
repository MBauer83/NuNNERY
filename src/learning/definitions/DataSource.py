from abc import ABC, abstractmethod
import numpy as np
from typing import Generic, TypeVar

PermittedDTypes = int|float|tuple[int|float,...]

DataSourceType_co = TypeVar('DataSourceType_co', covariant=True, bound='PermittedDTypes')
class DataSource(Generic[DataSourceType_co], ABC):

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_next_batch(self, size: int = 1) -> np.ndarray[DataSourceType_co]|None:
        raise NotImplementedError()
    
    @abstractmethod
    def get_all_data(self) -> np.ndarray[DataSourceType_co]:
        raise NotImplementedError()
    
    @abstractmethod
    def reset(self) -> None:
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
                callable(subclass.reset)) or
                NotImplemented)