from typing import *
import numpy as np

class Weights:
    def as_array() -> np.ndarray[float]:
        pass

    def add(self, weights_delta: np.ndarray[float]) -> None:
        pass

    def multiply(self, factor: float) -> None:
        pass

    @staticmethod
    def generate(shape: tuple[int, int], initializer: Callable[[int], np.ndarray[float]]) -> 'Weights':
        pass

    