import cv2
from src.ui.layout import Rect

class SeparatorComponent:
    def draw(self, frame, rect: Rect, color: tuple, thickness: int = 2):
        # Separators are obsolete in the new glassmorphic floating card layout.
        # Whitespace and card gaps define the layout.
        pass

