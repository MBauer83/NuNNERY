from typing import *
import numpy as np
from learning.definitions.ClassificationMetrics import ClassificationMetrics
from .BinaryClassificationMetrics import BinaryClassificationMetrics
from .ClassificationMetricsMixin import ClassificationMetricsMixin

class MultiClassClassificationMetrics(ClassificationMetrics, ClassificationMetricsMixin):

    def __init__(self, class_names: List[str], confusion_matrix: np.ndarray[int|float], per_class_metrics: List[BinaryClassificationMetrics]):
        self.__multiclass_confusion_matrix = confusion_matrix
        self.__multiclass_rates_matrix = self.__get_multiclass_rates_matrix(confusion_matrix)
        self.__per_class_metrics = per_class_metrics

        # sum up true positives, false positives, true negatives, false negatives over the per_class_metrics
        total_true_positives = np.sum([metrics.get_no_of_true_positives() for metrics in per_class_metrics])
        total_false_positives = np.sum([metrics.get_no_of_false_positives() for metrics in per_class_metrics])
        total_true_negatives = np.sum([metrics.get_no_of_true_negatives() for metrics in per_class_metrics])
        total_false_negatives = np.sum([metrics.get_no_of_false_negatives() for metrics in per_class_metrics])

        # call the super class's init method to set standard values
        super().init(class_names, total_true_positives, total_false_positives, total_true_negatives, total_false_negatives)

        # override the accuracy, precision, recall, f1_score, and support values with weighted averages
        no_of_data_points = np.sum(confusion_matrix)
        (
            self.__accuracy,
            self.__precision,
            self.__recall,
            self.__f1_score,
            self.__support
        ) = self.__calculate_weighted_averages_from_per_class_metrics(no_of_data_points, per_class_metrics)


    def _get_multiclass_rates_matrix(self, confusion_matrix: np.ndarray) -> np.ndarray:
        # sum over all values in a column to get the total number of samples in that class
        # divide each value in a column by the total number of samples in that class to get the rate for that class
        rates_matrix = np.ndarray(confusion_matrix.shape, dtype=float)
        for i in range(len(confusion_matrix)):
            column = confusion_matrix[:, i]
            class_total = np.sum(column)
            rates_matrix[:, i] = column / class_total

    def _calculate_weighted_averages_from_per_class_metrics(self, no_of_data_points: int, per_class_metrics: np.ndarray) -> float:
        avg_accuracy = 0.
        avg_precision = 0.
        avg_recall = 0.
        avg_f1_score = 0.
        avg_support = 0.
        for i in range(len(per_class_metrics)):
            metrics: BinaryClassificationMetrics = per_class_metrics[i]
            no_of_samples_in_class = metrics.get_support()
            weight = no_of_samples_in_class / no_of_data_points
            avg_accuracy += weight * metrics.get_accuracy()
            avg_precision += weight * metrics.get_precision()
            avg_recall += weight * metrics.get_recall()
            avg_f1_score += weight * metrics.get_f1_score()
            avg_support += weight * metrics.get_support()
        return avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_support
    
    def print(self):
        joined_class_names = ' | '.join(self.__class_names)
        print()
        print()
        print(
            f'Classification metrics for multi-class problem ({joined_class_names}):'
        )
        print(
            f'True positives: {self.__true_positives} ({self.__true_positive_rate * 100:.4f}%)'
        )
        print(
            f'False positives: {self.__false_positives} ({self.__false_positive_rate * 100:.4f}%)'
        )
        print(
            f'True negatives: {self.__true_negatives} ({self.__true_negative_rate * 100:.4f}%)'
        )
        print(
            f'False negatives: {self.__false_negatives} ({self.__false_negative_rate * 100:.4f}%)'
        )
        print(f'Accuracy: {self.__accuracy:.4f}')
        print(f'Precision: {self.__precision:.4f}')
        print(f'Recall: {self.__recall:.4f}')
        print(f'F1 score: {self.__f1_score:.4f}')
        print(f'Support: {self.__support:.4f}')
        print()
        print("Confusion matrix:")
        print(
            self.__format_confusion_matrix(
                self.__format_confusion_matrix_data(
                    self.__multiclass_confusion_matrix, self.__multiclass_rates_matrix
                )
            )
        )
    

    @staticmethod
    def from_raw_data(class_names: List[str], expected_one_hot: List[np.ndarray[int]], actual_one_hot: List[np.ndarray[int]]) -> 'ClassificationMetrics':
        # check dimensions of expected and actual match
        if len(expected_one_hot) != len(actual_one_hot):
            raise ValueError("Expected and actual data must have the same length")
        if len(expected_one_hot[0]) != len(actual_one_hot[0]):
            raise ValueError("Expected and actual data must have the same number of columns")
        if len(class_names) != len(expected_one_hot[0]):
            raise ValueError("Expected class names to match number of columns in expected data")
        
        # Calculate individual metrics as well as overall confusion matrix for a multi-class classification problem.
        # The matrix has a column for each actual class and a row for each predicted class.
        # Each cell contains the number of samples that were predicted to be in the row's class and were actually in the column's class.
        per_class_metrics: np.ndarray[BinaryClassificationMetrics] = np.empty(len(class_names), dtype=BinaryClassificationMetrics)    
        confusion_matrix = np.ndarray((len(class_names), len(class_names)), dtype=int)
        for i in range(len(class_names)):
            expected = expected_one_hot[:, i]
            actual = actual_one_hot[:, i]
            class_i_metrics = BinaryClassificationMetrics.from_raw_data(expected, actual)
            per_class_metrics[i] = class_i_metrics
            for j in range(len(class_names)):
                expected_j = expected_one_hot[:, j]
                actual_i = actual_one_hot[:, i]
                confusion_matrix[i, j] = np.sum(np.logical_and(expected_j, actual_i))

        return MultiClassClassificationMetrics(class_names, confusion_matrix, per_class_metrics)

