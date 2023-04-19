from typing import *
import numpy as np

class ClassificationMetricsMixin:

    def __init__(self, class_names: List[str], true_positives: int, false_positives: int, true_negatives: int, false_negatives: int):
        self.__class_names = class_names
        self.__true_positives = true_positives
        self.__false_positives = false_positives
        self.__true_negatives = true_negatives
        self.__false_negatives = false_negatives

        self.__true_positive_rate = true_positives / (true_positives + false_negatives)
        self.__false_positive_rate = false_positives / (false_positives + true_negatives)
        self.__true_negative_rate = true_negatives / (true_negatives + false_positives)
        self.__false_negative_rate = false_negatives / (false_negatives + true_positives)

        self.__overall_confusion_matrix = np.array([
            [true_positives, false_positives],
            [false_negatives, true_negatives]
        ])

        self.__overall_rates_matrix = np.array([
            [self.__true_positive_rate, self.__false_positive_rate],
            [self.__false_negative_rate, self.__true_negative_rate]
        ])

        total = true_positives + false_positives + true_negatives + false_negatives
        self.__accuracy = (true_positives + true_negatives) / total
        self.__precision = true_positives / (true_positives + false_positives)
        self.__recall = true_positives / (true_positives + false_negatives)
        self.__f1_score = 2 * (self.__precision * self.__recall) / (self.__precision + self.__recall)
        self.__support = total



    def get_class_names(self) -> List[str]:
        return self.__class_names
    def get_no_of_true_positives(self) -> int:
        return self.__true_positives
    def get_no_of_false_positives(self) -> int:
        return self.__false_positives
    def get_no_of_true_negatives(self) -> int:
        return self.__true_negatives
    def get_no_of_false_negatives(self) -> int:
        return self.__false_negatives
    def get_true_positive_rate(self) -> float:
        return self.__true_positive_rate
    def get_false_positive_rate(self) -> float:
        return self.__false_positive_rate
    def get_true_negative_rate(self) -> float:
        return self.__true_negative_rate
    def get_false_negative_rate(self) -> float:
        return self.__false_negative_rate
    def get_accuracy(self) -> float|int:
        return self.__accuracy
    def get_precision(self) -> float|int:
        return self.__precision
    def get_recall(self) -> float|int:
        return self.__recall
    def get_f1_score(self) -> float|int:
        return self.__f1_score
    def get_support(self) -> float|int:
        return self.__support
    def get_overall_confusion_matrix(self) -> np.ndarray[int]:
        return self.__overall_confusion_matrix
    def get_overall_rates_matrix(self) -> np.ndarray[float]:
        return self.__overall_rates_matrix
        
    def as_array(self) -> np.ndarray[float|int]:
        return np.array([
            self.__true_positives,
            self.__false_positives,
            self.__true_negatives,
            self.__false_negatives,
            self.__true_positive_rate,
            self.__false_positive_rate,
            self.__true_negative_rate,
            self.__false_negative_rate,
            self.__accuracy,
            self.__precision,
            self.__recall,
            self.__f1_score,
            self.__support
        ])
    
    def as_dict(self) -> Dict[str, float|int]:
        return {
            'true_positives': self.__true_positives,
            'false_positives': self.__false_positives,
            'true_negatives': self.__true_negatives,
            'false_negatives': self.__false_negatives,
            'true_positive_rate': self.__true_positive_rate,
            'false_positive_rate': self.__false_positive_rate,
            'true_negative_rate': self.__true_negative_rate,
            'false_negative_rate': self.__false_negative_rate,
            'accuracy': self.__accuracy,
            'precision': self.__precision,
            'recall': self.__recall,
            'f1_score': self.__f1_score,
            'support': self.__support
        }

    def get_class_names(self) -> List[str]:
        return self.__class_names

    def calculate_accuracy(self, true_positives: int, false_positives: int, true_negatives: int, false_negatives: int) -> float:
        return (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    
    def calculate_precision(self, true_positives: int, false_positives: int) -> float:
        return true_positives / (true_positives + false_positives)
    
    def calculate_recall(self, true_positives: int, false_negatives: int) -> float:
        return true_positives / (true_positives + false_negatives)
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        return 2 * precision * recall / (precision + recall)
    
    def calculate_support(self, true_positives: int, false_negatives: int) -> int:
        return true_positives + false_negatives
    
    def _calculate_overall_rates_matrix(self, true_positives: int, false_positives: int, true_negatives: int, false_negatives: int) -> np.ndarray:
        total = true_positives + false_positives + true_negatives + false_negatives
        true_positive_rate = true_positives / total
        false_positive_rate = false_positives / total
        true_negative_rate = true_negatives / total
        false_negative_rate = false_negatives / total
        return \
            np.array([
                [true_positive_rate, false_positive_rate],
                [false_negative_rate, true_negative_rate]
            ])
    
    def _format_confusion_matrix_data(self, confusion_matrix: np.ndarray, rates_matrix: np.ndarray) -> np.ndarray:
        f = lambda x, y: f'{confusion_matrix[x][y]} ({rates_matrix[x][y] * 100:.2f}%)'
        formatted_matrix_data = np.empty(confusion_matrix.shape, dtype=str)
        for x in len(confusion_matrix):
            for y in len(confusion_matrix[x]):
                formatted_matrix_data[x][y] = f(x, y)
        return formatted_matrix_data
    
    def _format_confusion_matrix(self, formatted_confusion_matrix_data: np.ndarray) -> str:
        # print confusion matrix with header labels and row labels
        class_names_list = list(self.__class_names)
        longest_class_name_length = max([len(x) for x in class_names_list])
        # longest field length is max string length from printable confusion matrix
        longest_field_length = np.max([len(x) for x in formatted_confusion_matrix_data.flatten()])
        padding_length = max(longest_field_length, longest_class_name_length) + 1
        get_padding_length_l = lambda string: int((padding_length - len(string)) / 2)
        get_padding_length_r = lambda string: padding_length - get_padding_length_l(string)
        pad_str_left = np.vectorize(lambda s: f'{" " * (padding_length - len(s))}{s}')
        pad_string_center = np.vectorize(lambda s: f'{" " * get_padding_length_l(s)}{s}{" " * get_padding_length_r(s)}')
    
        lpadded_class_names = pad_str_left(class_names_list)
        cpadded_class_names = pad_string_center(class_names_list)
        # map the printable_confusion_matrix to a new matrix with padded strings
        padded_printable_matrix = pad_str_left(formatted_confusion_matrix_data)
        # create header with centrally padded class-names
        header = f'{" " * padding_length}' + ''.join([f'|{x}' for x in cpadded_class_names]) + '\n'
        rows = []
        for i, row in enumerate(padded_printable_matrix):
            row = f'{lpadded_class_names[i]}' + ''.join([f'|{x}' for x in row])
            rows.append(row + '\n')
        return header + ''.join(rows)