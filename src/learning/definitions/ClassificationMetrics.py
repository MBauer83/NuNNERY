from typing import *
import numpy as np

class ClassificationMetrics:

    def get_class_names(self) -> List[str]:
        pass
    def get_no_of_true_positives(self) -> int:
        pass
    def get_no_of_false_positives(self) -> int:
        pass
    def get_no_of_true_negatives(self) -> int:
        pass
    def get_no_of_false_negatives(self) -> int:
        pass
    def get_true_positive_rate(self) -> float:
        pass
    def get_false_positive_rate(self) -> float:
        pass
    def get_true_negative_rate(self) -> float:
        pass
    def get_false_negative_rate(self) -> float:
        pass
    def get_accuracy(self) -> float|int:
        pass
    def get_precision(self) -> float|int:
        pass
    def get_recall(self) -> float|int:
        pass
    def get_f1_score(self) -> float|int:
        pass
    def get_support(self) -> float|int:
        pass
    def get_overall_confusion_matrix(self) -> np.ndarray[int]:
        pass
    def get_overall_rates_matrix(self) -> np.ndarray[float]:
        pass
    def as_array(self) -> np.ndarray[float|int]:
        pass
    def as_dict(self) -> Dict[str, float|int]:
        pass
    def print(self):
        pass
    @staticmethod
    def from_raw_data(class_names: List[str], expected_one_hot: List[np.ndarray[int]], actual_one_hot: List[np.ndarray[int]]) -> 'ClassificationMetrics':
        pass