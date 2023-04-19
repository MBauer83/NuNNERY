# append base directory `./../../`
import sys
# get dirname of this file
sys.path.append('./../../')



# write a test which creates a small DefaultNeuralNetwork using its generate method
# then generates some small input data and calls forward on the neural network
# checking that the output is not all zeros
from src.core.implementations.DefaultNeuralNetwork import DefaultNeuralNetwork
from src.core.implementations.ReLU import ReLU
from src.core.implementations.Softmax import Softmax
import numpy as np

def test_forward():
    network = DefaultNeuralNetwork.generate((2, 2, 2), [ReLU, Softmax])
    input = np.array([1, 2])
    output = network.forward(input)
    assert not np.all(output == 0)
    print(f'Output: {output}')

test_forward()
