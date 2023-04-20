import numpy as np
from abc import ABC, abstractmethod

class ClassificationMetrics(ABC):

    @abstractmethod
    def get_class_names(self) -> list[str]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_no_of_true_positives(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_no_of_false_positives(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_no_of_true_negatives(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_no_of_false_negatives(self) -> int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_true_positive_rate(self) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def get_false_positive_rate(self) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def get_true_negative_rate(self) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def get_false_negative_rate(self) -> float:
        raise NotImplementedError()
    
    @abstractmethod
    def get_accuracy(self) -> float|int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_precision(self) -> float|int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_recall(self) -> float|int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_f1_score(self) -> float|int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_support(self) -> float|int:
        raise NotImplementedError()
    
    @abstractmethod
    def get_overall_confusion_matrix(self) -> np.ndarray[int]:
        raise NotImplementedError()
    
    @abstractmethod
    def get_overall_rates_matrix(self) -> np.ndarray[float]:
        raise NotImplementedError()
    
    @abstractmethod
    def as_array(self) -> np.ndarray[float|int]:
        raise NotImplementedError()
    
    @abstractmethod
    def print(self):
        raise NotImplementedError()
    
    @staticmethod
    def from_raw_data(class_names: list[str], expected_one_hot: list[np.ndarray[int]], actual_one_hot: list[np.ndarray[int]]) -> 'ClassificationMetrics':
        raise NotImplementedError()
    
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get_class_names') and 
                callable(subclass.get_class_names) and
                hasattr(subclass, 'get_no_of_true_positives') and
                callable(subclass.get_no_of_true_positives) and
                hasattr(subclass, 'get_no_of_false_positives') and
                callable(subclass.get_no_of_false_positives) and
                hasattr(subclass, 'get_no_of_true_negatives') and
                callable(subclass.get_no_of_true_negatives) and
                hasattr(subclass, 'get_no_of_false_negatives') and
                callable(subclass.get_no_of_false_negatives) and
                hasattr(subclass, 'get_true_positive_rate') and
                callable(subclass.get_true_positive_rate) and
                hasattr(subclass, 'get_false_positive_rate') and
                callable(subclass.get_false_positive_rate) and
                hasattr(subclass, 'get_true_negative_rate') and
                callable(subclass.get_true_negative_rate) and
                hasattr(subclass, 'get_false_negative_rate') and
                callable(subclass.get_false_negative_rate) and
                hasattr(subclass, 'get_accuracy') and
                callable(subclass.get_accuracy) and
                hasattr(subclass, 'get_precision') and
                callable(subclass.get_precision) and
                hasattr(subclass, 'get_recall') and
                callable(subclass.get_recall) and
                hasattr(subclass, 'get_f1_score') and
                callable(subclass.get_f1_score) and
                hasattr(subclass, 'get_support') and
                callable(subclass.get_support) and
                hasattr(subclass, 'get_overall_confusion_matrix') and
                callable(subclass.get_overall_confusion_matrix) and
                hasattr(subclass, 'get_overall_rates_matrix') and
                callable(subclass.get_overall_rates_matrix) and
                hasattr(subclass, 'as_array') and
                callable(subclass.as_array) and
                hasattr(subclass, 'print') and
                callable(subclass.print) and
                hasattr(subclass, 'from_raw_data') and
                callable(subclass.from_raw_data)) or NotImplemented
    