from dataclasses import dataclass
from typing import Dict

from src.ui.constants import Constants


@dataclass
class Rect:
    """Represents a rectangle on the screen."""
    x: int
    y: int
    w: int
    h: int


class LayoutManager:
    """
    Professional Dashboard Layout Manager

    Responsibilities:
    - Calculates all floating panel positions.
    - Keeps webcam as the primary focus.
    - Uses responsive sizing.
    """

    def __init__(self):
        self.rects: Dict[str, Rect] = {}

    def calculate_layout(self, frame_width: int, frame_height: int) -> None:
        """
        Calculate the layout for every floating UI card.
        """

        margin_x = Constants.MARGIN_X
        margin_y = Constants.MARGIN_Y
        gap = Constants.GAP

        # ==========================================================
        # HEADER
        # ==========================================================

        header_width = min(
            int(frame_width * 0.55),
            frame_width - (2 * margin_x)
        )

        header_x = (frame_width - header_width) // 2

        self.rects["header"] = Rect(
            x=header_x,
            y=margin_y,
            w=header_width,
            h=Constants.HEADER_HEIGHT,
        )

        # ==========================================================
        # BOTTOM DASHBOARD ROW (Prediction | Sentence)
        # ==========================================================

        row_y = (
            frame_height
            - margin_y
            - Constants.DATASET_PRED_SENT_HEIGHT
        )

        available_width = (
            frame_width
            - (2 * margin_x)
            - gap
        )

        prediction_width = int(
            available_width * Constants.PREDICTION_WIDTH_PCT
        )

        sentence_width = (
            available_width
            - prediction_width
        )

        # ==========================================================
        # PREDICTION CARD
        # ==========================================================

        prediction_x = margin_x

        self.rects["prediction"] = Rect(
            x=prediction_x,
            y=row_y,
            w=prediction_width,
            h=Constants.DATASET_PRED_SENT_HEIGHT,
        )

        # ==========================================================
        # SENTENCE CARD
        # ==========================================================

        sentence_x = (
            prediction_x
            + prediction_width
            + gap
        )

        self.rects["sentence"] = Rect(
            x=sentence_x,
            y=row_y,
            w=sentence_width,
            h=Constants.DATASET_PRED_SENT_HEIGHT,
        )

    def get_rect(self, name: str) -> Rect:
        """
        Return the rectangle for a UI component.
        """
        return self.rects.get(name, Rect(0, 0, 0, 0))
        # ==========================================================
    # UI TOOLKIT HELPERS (Version 2)
    # ==========================================================

    @classmethod
    def draw_text(
        cls,
        img,
        text,
        x,
        y,
        scale=None,
        color=None,
        thickness=None,
        font=None,
    ):
        """Draw text with theme defaults."""

        if scale is None:
            scale = cls.BODY

        if color is None:
            color = cls.WHITE

        if thickness is None:
            thickness = cls.THICKNESS_NORMAL

        if font is None:
            font = cls.FONT_SECONDARY

        cv2.putText(
            img,
            str(text),
            (int(x), int(y)),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # ----------------------------------------------------------

    @classmethod
    def draw_centered_text(
        cls,
        img,
        text,
        center_x,
        center_y,
        scale=None,
        color=None,
        thickness=None,
        font=None,
    ):
        """Draw text centered around a point."""

        if scale is None:
            scale = cls.BODY

        if color is None:
            color = cls.WHITE

        if thickness is None:
            thickness = cls.THICKNESS_NORMAL

        if font is None:
            font = cls.FONT_SECONDARY

        (w, h), _ = cv2.getTextSize(
            str(text),
            font,
            scale,
            thickness,
        )

        x = int(center_x - w / 2)
        y = int(center_y + h / 2)

        cls.draw_text(
            img,
            text,
            x,
            y,
            scale,
            color,
            thickness,
            font,
        )

    # ----------------------------------------------------------

    @classmethod
    def draw_card_title(cls, img, rect: Rect, title):
        """Standard title used by every card."""

        cls.draw_text(
            img,
            title.upper(),
            rect.x + 20,
            rect.y + 28,
            scale=cls.H2,
            color=cls.ACCENT,
            thickness=cls.THICKNESS_NORMAL,
            font=cls.FONT_PRIMARY,
        )

    # ----------------------------------------------------------

    @classmethod
    def draw_divider(
        cls,
        img,
        x1,
        y,
        x2,
        color=None,
        thickness=1,
    ):
        """Draw a subtle divider."""

        if color is None:
            color = cls.SECONDARY

        cv2.line(
            img,
            (int(x1), int(y)),
            (int(x2), int(y)),
            color,
            thickness,
            cv2.LINE_AA,
        )

    # ----------------------------------------------------------

    @classmethod
    def draw_progress_bar(
        cls,
        img,
        x,
        y,
        width,
        height,
        progress,
        fill_color=None,
    ):
        """Draw a rounded progress bar."""

        if fill_color is None:
            fill_color = cls.SUCCESS

        progress = max(0.0, min(1.0, progress))

        cls.draw_rounded_rect(
            img,
            (x, y),
            (x + width, y + height),
            cls.PROGRESS_BG,
            -1,
            height // 2,
        )

        filled = int(width * progress)

        if filled > 0:
            cls.draw_rounded_rect(
                img,
                (x, y),
                (x + filled, y + height),
                fill_color,
                -1,
                min(height // 2, max(1, filled // 2)),
            )

    # ----------------------------------------------------------

    @classmethod
    def draw_chip(
        cls,
        img,
        x,
        y,
        text,
        bg_color,
        text_color=None,
        padding_x=10,
        padding_y=5,
        scale=None,
    ):
        """Draw a rounded badge/chip."""

        if text_color is None:
            text_color = cls.BLACK

        if scale is None:
            scale = cls.CAPTION

        (tw, th), _ = cv2.getTextSize(
            str(text),
            cls.FONT_SECONDARY,
            scale,
            cls.THICKNESS_NORMAL,
        )

        width = tw + padding_x * 2
        height = th + padding_y * 2

        cls.draw_rounded_rect(
            img,
            (x, y),
            (x + width, y + height),
            bg_color,
            -1,
            8,
        )

        cls.draw_text(
            img,
            text,
            x + padding_x,
            y + height - padding_y - 2,
            scale=scale,
            color=text_color,
            thickness=cls.THICKNESS_NORMAL,
        )

        return width, height

    # ----------------------------------------------------------

    @classmethod
    def draw_status_chip(
        cls,
        img,
        x,
        y,
        status,
    ):
        """Draw status badge with automatic coloring."""

        status_upper = str(status).upper()

        if status_upper in ["STATIC"]:
            color = cls.ACCENT

        elif status_upper in [
            "READY",
            "SAVED",
            "COMPLETE",
            "DYNAMIC",
            "TRACKING",
        ]:
            color = cls.SUCCESS

        elif status_upper in [
            "MOVE",
            "RECORDING",
            "WAIT",
        ]:
            color = cls.RECORDING

        elif status_upper in [
            "ERROR",
            "FAILED",
        ]:
            color = cls.ERROR

        else:
            color = cls.SECONDARY

        return cls.draw_chip(
            img,
            x,
            y,
            status_upper,
            color,
        )

    # ----------------------------------------------------------

    @classmethod
    def draw_center_value(
        cls,
        img,
        value,
        center_x,
        center_y,
        color=None,
        scale=2.0,
    ):
        """Large centered value."""

        if color is None:
            color = cls.WHITE

        cls.draw_centered_text(
            img,
            str(value),
            center_x,
            center_y,
            scale=scale,
            color=color,
            thickness=cls.THICKNESS_THICK,
            font=cls.FONT_PRIMARY,
        )