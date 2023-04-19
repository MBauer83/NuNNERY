from typing import *
import numpy as np
from .LossFunction import LossFunction
from .Classification import Classification
from .NeuralNetworkPerformanceStatistics import NeuralNetworkPerformanceStatistics

class NeuralNetworkProblem:
    def get_training_validation_test_split() -> tuple[float, float, float]:
        pass
    def get_loss_function() -> LossFunction:
        pass
    def get_classification() -> Classification|None:
        pass
    def measure_performance(self, expected_output: np.ndarray, actual_output: np.ndarray) -> NeuralNetworkPerformanceStatistics:
        pass
