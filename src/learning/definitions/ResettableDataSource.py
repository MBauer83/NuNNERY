from abc import ABC, abstractmethod
from typing import Generic
from .DataSource import DataSourceType_co, DataSource

class ResettableDataSource(Generic[DataSourceType_co], DataSource[DataSourceType_co], ABC):
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
                callable(subclass.reset) and
                hasattr(subclass, 'reset') and
                callable(subclass.reset)) or
                NotImplemented)