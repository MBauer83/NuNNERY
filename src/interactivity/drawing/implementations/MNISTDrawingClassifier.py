import numpy as np
import cv2
import matplotlib.pyplot as plt
from ....core.definitions.NeuralNetwork import NeuralNetwork
from ..definitions.OnLineDrawing import OnLineDrawing
from ..value_objects.LineDrawingConfiguration import LineDrawingConfiguration

class MNISTDrawingClassifier:
    def __initialize(self, nn: NeuralNetwork, drawing: OnLineDrawing):
        self._nn: NeuralNetwork = nn
        self._drawing: OnLineDrawing = drawing
        self._input_shape: tuple[int, ...] = self._nn.shape()[0],
        self._input_length: int = np.product(self._input_shape)
        self._orig_side_length: int = int(np.sqrt(self._input_length))
        self._config: LineDrawingConfiguration = self._get_config()

    def _get_config(self) -> LineDrawingConfiguration:
        orig_side_length = self._orig_side_length
        drawing_side_length_multiplier = 1

        # Multiply the square dimensions until we are above 600px
        drawing_side_length = orig_side_length
        while drawing_side_length < 600:
            drawing_side_length_multiplier += 1
            drawing_side_length = orig_side_length * drawing_side_length_multiplier

        # Create the configuration
        bg_color = (0, 0, 0)  # Black background
        line_color = (255, 255, 255)  # White line color
        line_width = int(0.08 * drawing_side_length)  # Line width is 8% of side-length
        return LineDrawingConfiguration(
            height=drawing_side_length,
            width=drawing_side_length,
            bg_color=bg_color,
            line_color=line_color,
            line_width=line_width
        )

    def run(self, nn: NeuralNetwork, drawing: OnLineDrawing) -> None:
        self.__initialize(nn, drawing)

        # Get compiled network function
        network_fn = self._nn.compile()

        # Run the drawing generator
        for image in self._drawing.run(self._config):
            # Resize the image
            resized_image = cv2.resize(
                image, (self._orig_side_length,) * 2, interpolation=cv2.INTER_AREA
            )

            # Get the class probabilities
            class_probs = network_fn(resized_image.flatten())
            
            self.__update_output(class_probs)


    @staticmethod
    def __update_output(class_probs: np.ndarray) -> None:
            # Create bar graph of class probabilities
            classes = list(range(10))
            _, ax = plt.subplots()
            ax.bar(classes, class_probs)
            ax.set_xlabel('Class')
            ax.set_ylabel('Probability')
            ax.set_xticks(classes)
            ax.set_xticklabels(classes)
            ax.set_ylim([0, 1])
            plt.show(block=False)
            plt.pause(0.001)
            plt.clf()