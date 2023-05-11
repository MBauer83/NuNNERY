import numpy as np
import cv2
from typing import Generator
import sys
sys.path.insert(0, "..")
from ..definitions.OnLineDrawing import OnLineDrawing
from ..value_objects.LineDrawingConfiguration import LineDrawingConfiguration

class CV2OnLineDrawingGenerator(OnLineDrawing):

    def __init__(self):
        self._color_dim = None
        self._send = None
        self._quit = None
        self._last_point = None
        self._drawing = None
        self._config = None

    def run(self, config: LineDrawingConfiguration) -> Generator[np.ndarray, None, None]:
        print('run called')
        self._config: LineDrawingConfiguration = config
        self._drawing: bool = False
        self._last_point: tuple[int, int] = (0, 0)
        self._send: bool = False
        self._quit: bool = False
        # if bg_color is black and line_color is white or vice versa
        # the third dimension is 1, otherwise 3
        bg_color = config.bg_color
        line_color = config.line_color
        self._color_dim: int = \
            1 if (bg_color == (0,0,0) and line_color == (255,255,255)) or \
                 (bg_color == (255,255,255) and line_color == (0,0,0)) \
            else 3
        self.__clear_canvas()
        print('Calling namedWindow')
        cv2.namedWindow("canvas")
        print('namedWindow called')
        cv2.setMouseCallback("canvas", self.__draw_circle)

        # create the "send", "clear", and "quit" buttons
        cv2.createButton("Send", self.__set_send)
        cv2.createButton("Clear", self.__clear_canvas)
        cv2.createButton("Quit", self.__set_quit)


        while True:
            cv2.imshow("canvas", self.__canvas)
            k = cv2.waitKey(33)
            if k == -1:
                i = 0
            elif k == 27:
                break
            if self._quit:
                break
            if self._send:
                yield self.__canvas
                self.__clear_canvas()
                self._send = False

        cv2.destroyAllWindows()

    def __draw_circle(self, event: int, x: int, y: int, _params: any, _flags: any) -> None:
        print(f'draw circle called')    
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            print(f'DRAWING LINE')
            cv2.line(self.__canvas, self._last_point, (x, y), self._config.line_color, self._config.line_width)
            self._last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False

    def __set_send(self, _event, _x, _y, _params, _flags) -> None:
        self._send = True

    def __clear_canvas(self, _event, _x, _y, _params, _flags) -> None:
        print('Clear canvas called.')
        self.__canvas: np.ndarray = np.zeros((self._config.width, self._config.height, self._color_dim), dtype=np.uint8)
        # fill with bg_color if not black
        if self._config.bg_color != (0, 0, 0):
            print(f'Filling with non-zero bg color: {self._config.bg_color}')
            self.__canvas.fill(self._config.bg_color)

    def __set_quit(self, _event, _x, _y, _params, _flags) -> None:
        self._quit = True        
