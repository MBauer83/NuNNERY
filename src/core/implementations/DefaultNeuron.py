from core.definitions import Neuron

class DefaultNeuron(Neuron):
    # bias need not be used if the bias-trick is used
    def __init__(self, activation: float, bias: float = 0.):
        self.activation = activation
        self.bias = bias

    def get_activation(self) -> float:
        return self.activation

    def get_bias(self) -> float:
        return self.bias