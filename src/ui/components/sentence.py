import cv2
from src.ui.layout import Rect
from src.ui.theme import Theme
from src.ui.constants import Constants


class SentenceComponent:
    """
    Sentence Display Component

    Shows:
    - Current sentence
    - Recently added letter
    """

    MAX_VISIBLE_CHARS = 34

    def draw(self, frame, rect: Rect, data: dict):

        # ---------------------------------------------------------
        # Background
        # ---------------------------------------------------------

        Theme.draw_glass_card(frame, rect)

        sentence = data.get("sentence", "")
        letter_added = data.get("letter_added", False)
        last_added_letter = data.get("last_added_letter", "")

        # ---------------------------------------------------------
        # Empty State
        # ---------------------------------------------------------

        if not sentence:
            sentence = "Waiting for signs..."

        # ---------------------------------------------------------
        # Prevent overflow
        # ---------------------------------------------------------

        if len(sentence) > self.MAX_VISIBLE_CHARS:
            sentence = "..." + sentence[-self.MAX_VISIBLE_CHARS:]

        pad = Constants.PADDING

        left = rect.x + pad
        right = rect.x + rect.w - pad

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------

        Theme.draw_card_title(
            frame,
            rect,
            "Sentence",
        )

        # ---------------------------------------------------------
        # Divider
        # ---------------------------------------------------------

        Theme.draw_divider(
            frame,
            left,
            rect.y + 42,
            right,
        )

        # ---------------------------------------------------------
        # Sentence Text
        # ---------------------------------------------------------

        Theme.draw_text(
            frame,
            sentence,
            left,
            rect.y + 82,
            scale=1.05,
            color=Theme.WHITE,
            thickness=Theme.THICKNESS_NORMAL,
            font=Theme.FONT_PRIMARY,
        )

        # ---------------------------------------------------------
        # Last Added Letter
        # ---------------------------------------------------------

        if letter_added and last_added_letter:

            chip_text = f"+ {last_added_letter}"

            

            # Estimate chip width
            (tw, th), _ = cv2.getTextSize(
                chip_text,
                Theme.FONT_SECONDARY,
                Theme.CAPTION,
                Theme.THICKNESS_NORMAL,
            )

            chip_width = tw + 20

            chip_x = right - chip_width
            chip_y = rect.y + rect.h - 38

            Theme.draw_chip(
                frame,
                chip_x,
                chip_y,
                chip_text,
                Theme.SUCCESS,
            )