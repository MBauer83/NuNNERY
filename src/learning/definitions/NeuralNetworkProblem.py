import numpy as np
from .LossFunction import LossFunction
from .Classification import Classification
from .LossStatistics import LossStatistics

class NeuralNetworkProblem:
    def get_training_validation_test_split(self) -> tuple[float, float, float]:
        pass
    def get_loss_function(self) -> LossFunction:
        pass
    def get_classification(self) -> Classification | None:
        pass
    def measure_performance(self, expected_output: np.ndarray, actual_output: np.ndarray) -> LossStatistics:
        pass
