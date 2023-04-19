import numpy as np
from typing import *
from src.core.definitions.Weights import Weights

class DefaultWeights(Weights):
    def __init__(self, weights: np.ndarray[float]):
        self.__weights = weights

    def get_weights(self) -> np.ndarray[float]:
        return self.weights

    def set_weights(self, weights: np.ndarray[float]) -> None:
        self.weights = weights

    def add(self, weights_delta: np.ndarray[float]) -> None:
        self.weights += weights_delta

    def multiply(self, factor: float) -> None:
        self.weights *= factor
    
    def as_array(self) -> np.ndarray[float]:
        return self.__weights
    
    def shape(self) -> tuple[int, ...]:
        return self.__weights.shape
    
    def __len__(self) -> int:
        return len(self.__weights)

    @staticmethod
    def generate(
        shape: tuple[int, int], 
        initializer: Callable[[tuple[int, int]], np.ndarray[float]] = \
            # He initialization
            lambda shape: np.random.normal(0., np.sqrt(2. / shape[0]), shape)            
    ) -> 'Weights':
        return DefaultWeights(initializer(shape))
