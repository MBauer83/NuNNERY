# append base directory `./../../`
import sys
sys.path.append('./../../')

import numpy as np
import unittest
from unittest.mock import MagicMock, patch

from src.core.definitions.NeuralNetwork import NeuralNetwork
from src.interactivity.drawing.definitions.OnLineDrawing import OnLineDrawing
from src.interactivity.drawing.implementations.MNISTDrawingClassifier import MNISTDrawingClassifier
from src.interactivity.drawing.value_objects.LineDrawingConfiguration import LineDrawingConfiguration


class TestMNISTDrawingClassifier(unittest.TestCase):

    def test_run(self):
        # Mock the NeuralNetwork and OnLineDrawing objects
        nn_mock = MagicMock(spec=NeuralNetwork)
        compiled_nn = lambda x: np.zeros((10,))
        nn_mock.shape.return_value = (784,256,128,10)
        nn_mock.compile.return_value = compiled_nn

        drawing_mock = MagicMock(spec=OnLineDrawing)
        drawing_mock.run.return_value = iter([
            np.zeros((616, 616, 1), dtype=np.uint8),
            np.ones((616, 616, 1), dtype=np.uint8)
        ])

        # Mock the MNISTDrawingClassifier object and call its run method
        classifier = MNISTDrawingClassifier()
        with patch.object(classifier, "_MNISTDrawingClassifier__update_output") as update_output_mock:
            classifier.run(nn_mock, drawing_mock)

        # Assert that the methods were called with the correct arguments
        nn_mock.compile.assert_called_once()
        expected_drawing_config = LineDrawingConfiguration(
            height=616,
            width=616,
            bg_color=(0, 0, 0),
            line_color=(255, 255, 255),
            line_width=49            
        )
        drawing_mock.run.assert_called_once_with(expected_drawing_config)
        self.assertEqual(len(update_output_mock.call_args_list), 2)
        
        # Assert that the update_output_mock was called with the correct arguments
        args, _ = update_output_mock.call_args_list[0]
        np.testing.assert_array_almost_equal(args[0], np.zeros(10))
        
        args, _ = update_output_mock.call_args_list[1]
        np.testing.assert_array_almost_equal(args[0], np.zeros(10))

if __name__ == '__main__':
    unittest.main()
