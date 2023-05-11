class LineDrawingConfiguration:
    def __init__(
        self, 
        height: int, 
        width: int, 
        bg_color: tuple[int, int, int], 
        line_color: tuple[int, int, int], 
        line_width: int
    ):
        self.__height: int = height
        self.__width: int = width
        self.__bg_color: tuple[int, int, int] = bg_color
        self.__line_color: tuple[int, int, int] = line_color
        self.__line_width: int = line_width
    
    @property
    def height(self) -> int:
        return self.__height
    
    @property
    def width(self) -> int:
        return self.__width
    
    @property
    def bg_color(self) -> tuple[int, int, int]:
        return self.__bg_color
    
    @property
    def line_color(self) -> tuple[int, int, int]:
        return self.__line_color
    
    @property
    def line_width(self) -> int:
        return self.__line_width
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LineDrawingConfiguration):
            return NotImplemented
        return self.height == other.height and \
            self.width == other.width and \
            self.bg_color == other.bg_color and \
            self.line_color == other.line_color and \
            self.line_width == other.line_width
    