from abc import ABC, abstractmethod
import numpy as np

class Classification(ABC):
    @abstractmethod
    def get_class_names(self) -> list[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_class_for_one_hot_encoding(self, one_hot_encoding: np.ndarray) -> str:
        raise NotImplementedError()
    
    @abstractmethod
    def get_classes_with_values_gte(self, output_activations: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_classes_with_values_lte(self, output_activations: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_classes_with_values_gt(self, output_activations: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_classes_with_values_lt(self, output_activations: np.ndarray, threshold: float) -> list[tuple[str, float]]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_one_hot_encoding(self, label: str) -> np.ndarray:
        raise NotImplementedError()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_class_names') and
                callable(subclass.get_class_names) and
                hasattr(subclass, 'get_class_for_one_hot_encoding') and
                callable(subclass.get_class_for_one_hot_encoding) and
                hasattr(subclass, 'get_classes_with_values_gte') and
                callable(subclass.get_classes_with_values_gte) and
                hasattr(subclass, 'get_classes_with_values_lte') and
                callable(subclass.get_classes_with_values_lte) and
                hasattr(subclass, 'get_classes_with_values_gt') and
                callable(subclass.get_classes_with_values_gt) and
                hasattr(subclass, 'get_classes_with_values_lt') and
                callable(subclass.get_classes_with_values_lt) and
                hasattr(subclass, 'get_one_hot_encoding') and
                callable(subclass.get_one_hot_encoding)) or NotImplemented