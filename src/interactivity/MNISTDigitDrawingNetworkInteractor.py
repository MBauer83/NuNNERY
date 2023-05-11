from ..core.definitions.NeuralNetwork import NeuralNetwork
from ..learning.definitions.NeuralNetworkProblem import NeuralNetworkProblem
from .NetworkInteractor import NetworkInteractor

# Uses TKinter open a window for drawing a digit and a window for displaying the output of the network. 
# Once a drawing event has taken place, it fetches the data from the canvas every 100ms while the user is drawing,
# passes it through the network and updates the second window with the output of the network.
class MNISTDigitDrawingNetworkInteractor(NetworkInteractor):
    def use_network(self, network: NeuralNetwork, problem: NeuralNetworkProblem):
        pass # TODO: Implement this method