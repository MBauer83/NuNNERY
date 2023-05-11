import numpy as np
from ...learning.definitions.ClassificationMetrics import ClassificationMetrics
from .ClassificationMetricsMixin import ClassificationMetricsMixin


class BinaryClassificationMetrics(ClassificationMetrics, ClassificationMetricsMixin):

    def __init__(
        self,
        class_names: tuple[str, str],
        true_positives: int,
        false_positives: int,
        true_negatives: int,
        false_negatives: int
    ):
        class_names_list = list(class_names)
        super().__init__(class_names_list, true_positives, false_positives, true_negatives, false_negatives)

    def print(self):
        print()
        print()
        print(
            f'Classification metrics for binary problem ({self.__class_names[0]} | {self.__class_names[1]}):'
        )
        print(
            f'True positives: {self.__true_positives}'
        )
        print(
            f'False positives: {self.__false_positives}'
        )
        print(
            f'True negatives: {self.__true_negatives}'
        )
        print(
            f'False negatives: {self.__false_negatives}'
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
                    self.__overall_confusion_matrix, self.__overall_rates_matrix
                )
            )
        )

    @staticmethod
    def _calculate_input_from_raw_data(
        expected_one_hot: list[np.ndarray[int]],
        actual_one_hot: list[np.ndarray[int]]
    ) -> tuple[int, int, int, int]:

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
        class_names: list[str],
        expected_one_hot: list[np.ndarray[int]],
        actual_one_hot: list[np.ndarray[int]]
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
