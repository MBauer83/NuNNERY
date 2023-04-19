from typing import *
import numpy as np

class NeuralNetworkPerformanceStatistics:
    def get_sum_losses(self) -> float:
        pass
    def get_mean_loss(self) -> float:
        pass
    def print(self):
        pass
    @staticmethod
    def from_float_array(array: np.ndarray[float]) -> 'NeuralNetworkPerformanceStatistics':
        pass
    @staticmethod
    def from_statistics(array: np.ndarray['NeuralNetworkPerformanceStatistics']) -> 'NeuralNetworkPerformanceStatistics':
        pass