import cv2
from src.ui.layout import Rect
from src.ui.theme import Theme

class ProgressBarComponent:
    def draw(self, frame, rect: Rect, percentage: float, label: str, bg_color: tuple, fill_color: tuple, text_color: tuple):
        """Draws a premium rounded progress bar."""
        # Cap percentage
        pct = max(0, min(100, percentage))
        
        # Background
        Theme.draw_rounded_rect(
            frame, 
            (rect.x, rect.y), 
            (rect.x + rect.w, rect.y + rect.h), 
            bg_color, 
            -1, 
            r=rect.h // 2
        )
        
        # Filled section
        filled_w = int((pct / 100.0) * rect.w)
        if filled_w > 0:
            Theme.draw_rounded_rect(
                frame, 
                (rect.x, rect.y), 
                (rect.x + filled_w, rect.y + rect.h), 
                fill_color, 
                -1, 
                r=min(rect.h // 2, filled_w // 2)
            )
            
        # Optional subtle inner glow/border
        Theme.draw_rounded_rect(
            frame, 
            (rect.x, rect.y), 
            (rect.x + rect.w, rect.y + rect.h), 
            Theme.SECONDARY, 
            1, 
            r=rect.h // 2
        )
        
        # Text
        if label:
            text_size = cv2.getTextSize(
                label,
                Theme.FONT_SECONDARY,
                Theme.BODY,
                Theme.THICKNESS_THIN
            )[0]
            
            text_x = rect.x + (rect.w - text_size[0]) // 2
            text_y = rect.y + (rect.h + text_size[1]) // 2 - 2
            
            cv2.putText(
                frame,
                label,
                (text_x, text_y),
                Theme.FONT_SECONDARY,
                Theme.BODY,
                text_color,
                Theme.THICKNESS_THIN,
                cv2.LINE_AA
            )
