# append base directory `./../../`
import sys
sys.path.append('./../../')

import cv2
import numpy as np
import unittest
from unittest.mock import patch

from src.interactivity.drawing.implementations.CV2OnLineDrawingGenerator import CV2OnLineDrawingGenerator
from src.interactivity.drawing.value_objects.LineDrawingConfiguration import LineDrawingConfiguration


class TestCV2OnLineDrawingGenerator(unittest.TestCase):

    def setUp(self):
        self.config = LineDrawingConfiguration(
            height=600,
            width=800,
            bg_color=(0, 0, 0),
            line_color=(255, 255, 255),
            line_width=2
        )
        self.generator = CV2OnLineDrawingGenerator()

    def test_run(self):
        # Mock the cv2 namedWindow and setMouseCallback functions
        #with patch.object(cv2, 'namedWindow') as namedWindow_mock, \
        #     patch.object(cv2, 'setMouseCallback') as setMouseCallback_mock:

        # Simulate drawing events and button clicks
        mock_canvas = np.zeros((self.config.width, self.config.height, 1), dtype=np.uint8)
        mock_canvas[0, 0] = 255
        self.generator._CV2OnLineDrawingGenerator__canvas = mock_canvas
        cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__draw_circle,
                                cv2.EVENT_LBUTTONDOWN, 100, 100)
        cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__draw_circle,
                                cv2.EVENT_MOUSEMOVE, 150, 150)
        cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__draw_circle,
                                cv2.EVENT_LBUTTONUP, 200, 200)
        cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__set_send,
                                cv2.EVENT_LBUTTONDOWN, 300, 300)
        cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__clear_canvas,
                                cv2.EVENT_LBUTTONDOWN, 400, 400)
        cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__set_quit,
                                cv2.EVENT_LBUTTONDOWN, 500, 500)
        
        print('Calling run')
        # Run the generator
        # iterate over generator, collect output
        drawings = self.generator.run(self.config)

        # Assert that the namedWindow and setMouseCallback functions were called with the correct arguments
        #namedWindow_mock.assert_called_once_with("canvas")
        #setMouseCallback_mock.assert_called_once_with("canvas", self.generator._CV2OnLineDrawingGenerator__draw_circle)


        # Test the yielded drawings
        equals = [mock_canvas, np.zeros((self.config.width, self.config.height, 1), dtype=np.uint8)]
        for i, canvas1 in enumerate(drawings):
            print(f'GOT DRAWING')
            np.testing.assert_array_equal(canvas1, equals[i])

    #def test_draw_circle(self):
    #    # Set up the drawing event
    #    event = cv2.EVENT_LBUTTONDOWN
    #    x, y = 100, 200
    #    self.generator._CV2OnLineDrawingGenerator__last_point = (50, 50)
    #    self.generator._CV2OnLineDrawingGenerator__drawing = False
#
    #    # Call the __draw_circle method
    #    self.generator._CV2OnLineDrawingGenerator__draw_circle(event, x, y)
#
    #    # Assert that the canvas was updated
    #    expected_canvas = np.zeros((self.config.width, self.config.height, 1), dtype=np.uint8)
    #    expected_canvas[50:201, 50:101] = self.config.line_color[0]
    #    np.testing.assert_array_equal(self.generator._CV2OnLineDrawingGenerator__canvas, expected_canvas)
#
    #    # Set up another drawing event
    #    event = cv2.EVENT_MOUSEMOVE
    #    x, y = 150, 250
    #    self.generator._CV2OnLineDrawingGenerator__last_point = (100, 200)
    #    self.generator._CV2OnLineDrawingGenerator__drawing = True
#
    #    # Call the __draw_circle method again
    #    self.generator._CV2OnLineDrawingGenerator__draw_circle(event, x, y)
#
    #    # Assert that the canvas was updated again
    #    expected_canvas[50:251, 50:101] = self.config.line_color[0]
    #    np.testing.assert_array_equal(self.generator._CV2OnLineDrawingGenerator__canvas, expected_canvas)
    #
    #def test_clear_canvas(self):
    #     with patch.object(cv2, 'namedWindow') as namedWindow_mock, \
    #         patch.object(cv2, 'setMouseCallback') as setMouseCallback_mock:
#
    #        # Run the generator
    #        drawings = self.generator.run(self.config)
#
    #        # Assert that the namedWindow and setMouseCallback functions were called with the correct arguments
    #        #namedWindow_mock.assert_called_once_with("canvas")
    #        #setMouseCallback_mock.assert_called_once_with("canvas", self.generator._CV2OnLineDrawingGenerator__clear_canvas)
#
    #        # Simulate clicking the clear button
    #        self.generator._CV2OnLineDrawingGenerator__canvas = np.ones((self.config.width, self.config.height, 1), dtype=np.uint8)
    #        cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__clear_canvas,
    #                            cv2.EVENT_LBUTTONDOWN)
    #        
#
    #        # Assert that the canvas is empty
    #        np.testing.assert_array_equal(self.generator._CV2OnLineDrawingGenerator__canvas, np.zeros((self.config.width, self.config.height, 1), dtype=np.uint8))
#
#
    #def test_send_canvas(self):
    #    # Simulate clicking the send button
    #    cv2.namedWindow("canvas")
    #    cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__set_send,
    #                            cv2.EVENT_LBUTTONDOWN, 0, 0)
    #    # Run the generator
    #    drawings = self.generator.run(self.config)
#
    #    # Assert that the yielded canvas is the same as the last drawn one
    #    canvas = next(drawings)
    #    np.testing.assert_array_equal(canvas, self.generator._CV2OnLineDrawingGenerator__canvas)
#
#
    #def test_quit_canvas(self):
    #    cv2.namedWindow("canvas")
    #    # Simulate clicking the quit button
    #    cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__set_quit,
    #                            cv2.EVENT_LBUTTONDOWN, 0, 0)
#
    #    # Run the generator
    #    drawings = self.generator.run(self.config)
#
    #    # Assert that the run method stops yielding drawings
    #    with self.assertRaises(StopIteration):
    #        next(drawings)
#
#
    #def test_draw_line(self):
    #    cv2.namedWindow("canvas")
    #    # Simulate drawing a line on the canvas
    #    cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__draw_circle,
    #                            cv2.EVENT_LBUTTONDOWN, 100, 200)
    #    cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__draw_circle,
    #                            cv2.EVENT_MOUSEMOVE, 150, 250)
    #    cv2.setMouseCallback("canvas", self.generator._CV2OnLineDrawingGenerator__draw_circle,
    #                            cv2.EVENT_LBUTTONUP, 200, 300)
#
    #    # Run the generator
    #    drawings = self.generator.run(self.config)
#
    #    # Assert that the yielded canvas is correct
    #    canvas = next(drawings)
    #    expected_canvas = np.zeros((self.config.width, self.config.height, 1), dtype=np.uint8)
    #    expected_canvas[50:301, 50:101] = self.config.line_color[0]
    #    np.testing.assert_array_equal(canvas, expected_canvas)
