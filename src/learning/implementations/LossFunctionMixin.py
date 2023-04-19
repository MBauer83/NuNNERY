import numpy as np

class LossFunctionMixin:
    @staticmethod
    def calculate(expected: np.ndarray, actual: np.ndarray) -> float:
        pass
    def __init__(self):
        self.__vectorized = np.vectorize(self.calculate)
     # can be useful for generating statistics after testing
    def vectorized(self, expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
        self.__vectorized(expected, actual)
