from typing import *
import numpy as np
from learning.definitions import ClassificationMetrics
from .ClassificationMetricsMixin import ClassificationMetricsMixin


class BinaryClassificationMetrics(ClassificationMetrics, ClassificationMetricsMixin):

    def __init__(
        self,
        class_names: Tuple[str, str],
        true_positives: int,
        false_positives: int,
        true_negatives: int,
        false_negatives: int
    ):
        super().__init__(class_names, true_positives, false_positives, true_negatives, false_negatives)

    def print(self):
        print()
        print()
        print(
            f'Classification metrics for binary problem ({self._class_names[0]} | {self._class_names[1]}):'
        )
        print(
            f'True positives: {self._true_positives} ({self._true_positives_rate * 100:.4f}%)'
        )
        print(
            f'False positives: {self._false_positives} ({self._false_positives_rate * 100:.4f}%)'
        )
        print(
            f'True negatives: {self._true_negatives} ({self._true_negatives_rate * 100:.4f}%)'
        )
        print(
            f'False negatives: {self._false_negatives} ({self._false_negatives_rate * 100:.4f}%)'
        )
        print(f'Accuracy: {self._accuracy:.4f}')
        print(f'Precision: {self._precision:.4f}')
        print(f'Recall: {self._recall:.4f}')
        print(f'F1 score: {self._f1_score:.4f}')
        print(f'Support: {self._support:.4f}')
        print()
        print("Confusion matrix:")
        print(
            self._format_confusion_matrix(
                self._format_confusion_matrix_data(
                    self._overall_confusion_matrix, self._overall_rates_matrix
                )
            )
        )

    @staticmethod
    def _calculate_input_from_raw_data(
        expected_one_hot: List[np.ndarray[int]],
        actual_one_hot: List[np.ndarray[int]]
    ) -> Tuple[int, int, int, int]:

        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for i in range(len(expected_one_hot)):
            if expected_one_hot[i][0] == 1 and actual_one_hot[i][0] == 1:
                true_positives += 1
            elif expected_one_hot[i][0] == 1 and actual_one_hot[i][1] == 1:
                false_negatives += 1
            elif expected_one_hot[i][1] == 1 and actual_one_hot[i][0] == 1:
                false_positives += 1
            elif expected_one_hot[i][1] == 1 and actual_one_hot[i][1] == 1:
                true_negatives += 1
            else:
                raise ValueError(
                    "Expected and actual data must be one-hot encoded")
        return true_positives, false_positives, true_negatives, false_negatives

    @staticmethod
    def from_raw_data(
        class_names: List[str],
        expected_one_hot: List[np.ndarray[int]],
        actual_one_hot: List[np.ndarray[int]]
    ) -> 'ClassificationMetrics':

        # check dimensions of expected and actual match
        if len(expected_one_hot) != len(actual_one_hot):
            raise ValueError(
                "Expected and actual data must have the same length")
        if len(expected_one_hot[0]) != len(actual_one_hot[0]):
            raise ValueError(
                "Expected and actual data must have the same number of columns")
        if len(expected_one_hot[0]) != 2:
            raise ValueError("Expected and actual data must have two columns")
        if len(class_names) != 2:
            raise ValueError("Expected two class names")
        # count true positives, false positives, true negatives, false negatives
        (
            true_positives,
            false_positives,
            true_negatives,
            false_negatives
        ) = BinaryClassificationMetrics._calculate_input_from_raw_data(expected_one_hot, actual_one_hot)

        return BinaryClassificationMetrics(
            class_names,
            true_positives,
            false_positives,
            true_negatives,
            false_negatives
        )
