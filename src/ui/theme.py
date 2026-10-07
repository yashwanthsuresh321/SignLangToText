import cv2
import numpy as np

from src.ui.layout import Rect


class Theme:
    """Professional UI Theme."""

    # ==========================================================
    # Colors (BGR)
    # ==========================================================

    BACKGROUND = (38, 37, 37)
    CARD_BG = (45, 45, 45)

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    ACCENT = (255, 229, 0)
    SUCCESS = (0, 220, 0)
    RECORDING = (0, 165, 255)
    ERROR = (0, 0, 255)
    SECONDARY = (160, 160, 160)

    PROGRESS_BG = (60, 60, 60)
    PROGRESS_FILL = ACCENT

    # ==========================================================
    # Typography
    # ==========================================================

    FONT_PRIMARY = cv2.FONT_HERSHEY_DUPLEX
    FONT_SECONDARY = cv2.FONT_HERSHEY_SIMPLEX

    H1 = 1.2
    H2 = 0.8
    BODY = 0.6
    CAPTION = 0.45

    THICKNESS_THIN = 1
    THICKNESS_NORMAL = 2
    THICKNESS_THICK = 3

    # ==========================================================
    # Layout
    # ==========================================================

    RADIUS_CARD = 12
    RADIUS_BUTTON = 6

    ALPHA_CARD = 0.75

    # ==========================================================
    # Status Colors
    # ==========================================================

    @classmethod
    def get_status_color(cls, status: str):

        s = status.lower()

        if "saved" in s:
            return cls.SUCCESS

        if "ready" in s:
            return cls.SUCCESS

        if "complete" in s:
            return cls.SUCCESS

        if "cooldown" in s:
            return cls.ACCENT

        if "move" in s:
            return cls.RECORDING

        if "record" in s:
            return cls.RECORDING

        return cls.WHITE

    # ==========================================================
    # Rounded Rectangle
    # ==========================================================

    @staticmethod
    def draw_rounded_rect(img, pt1, pt2, color, thickness, r):

        x1, y1 = pt1
        x2, y2 = pt2

        r = min(
            r,
            abs(x2 - x1) // 2,
            abs(y2 - y1) // 2,
        )

        if thickness < 0:

            cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1)

            cv2.rectangle(
                img,
                (x1 + r, y1),
                (x2 - r, y2),
                color,
                -1,
            )

            cv2.rectangle(
                img,
                (x1, y1 + r),
                (x2, y2 - r),
                color,
                -1,
            )

        else:

            cv2.ellipse(img, (x1 + r, y1 + r), (r, r),
                        180, 0, 90, color, thickness)

            cv2.ellipse(img, (x2 - r, y1 + r), (r, r),
                        270, 0, 90, color, thickness)

            cv2.ellipse(img, (x1 + r, y2 - r), (r, r),
                        90, 0, 90, color, thickness)

            cv2.ellipse(img, (x2 - r, y2 - r), (r, r),
                        0, 0, 90, color, thickness)

            cv2.line(
                img,
                (x1 + r, y1),
                (x2 - r, y1),
                color,
                thickness,
            )

            cv2.line(
                img,
                (x1 + r, y2),
                (x2 - r, y2),
                color,
                thickness,
            )

            cv2.line(
                img,
                (x1, y1 + r),
                (x1, y2 - r),
                color,
                thickness,
            )

            cv2.line(
                img,
                (x2, y1 + r),
                (x2, y2 - r),
                color,
                thickness,
            )

    # ==========================================================
    # Glass Card
    # ==========================================================

    @classmethod
    def draw_glass_card(cls, img, rect: Rect):

        overlay = img.copy()

        cls.draw_rounded_rect(
            overlay,
            (rect.x, rect.y),
            (rect.x + rect.w, rect.y + rect.h),
            cls.CARD_BG,
            -1,
            cls.RADIUS_CARD,
        )

        cls.draw_rounded_rect(
            overlay,
            (rect.x, rect.y),
            (rect.x + rect.w, rect.y + rect.h),
            cls.SECONDARY,
            1,
            cls.RADIUS_CARD,
        )

        cv2.addWeighted(
            overlay,
            cls.ALPHA_CARD,
            img,
            1 - cls.ALPHA_CARD,
            0,
            img,
        )

    # ==========================================================
    # Recording Dot
    # ==========================================================

    @classmethod
    def draw_recording_dot(
        cls,
        img,
        x,
        y,
        radius=6,
        is_recording=False,
    ):

        color = (
            cls.RECORDING
            if is_recording
            else cls.SECONDARY
        )

        cv2.circle(img, (x, y), radius, color, -1)

        if is_recording:

            cv2.circle(
                img,
                (x, y),
                radius + 4,
                color,
                1,
            )

    # ==========================================================
    # Check Icon
    # ==========================================================

    @classmethod
    def draw_check_icon(
        cls,
        img,
        x,
        y,
        size=10,
        color=None,
    ):

        if color is None:
            color = cls.SUCCESS

        pts = np.array(
            [
                [x, y],
                [x + size // 3, y + size // 3],
                [x + size, y - size // 2],
            ],
            np.int32,
        )

        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(
            img,
            [pts],
            False,
            color,
            cls.THICKNESS_NORMAL,
            cv2.LINE_AA,
        )

    # ==========================================================
    # Generic Text
    # ==========================================================

    @classmethod
    def draw_text(
        cls,
        img,
        text,
        x,
        y,
        scale,
        color,
        thickness=1,
        font=None,
    ):

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

    @classmethod
    def draw_centered_text(
        cls,
        img,
        text,
        center_x,
        y,
        scale,
        color,
        thickness=1,
        font=None,
    ):

        if font is None:
            font = cls.FONT_SECONDARY

        (w, _), _ = cv2.getTextSize(
            str(text),
            font,
            scale,
            thickness,
        )

        cls.draw_text(
            img,
            text,
            center_x - (w // 2),
            y,
            scale,
            color,
            thickness,
            font,
        )

    @classmethod
    def draw_center_value(
        cls,
        img,
        value,
        center_x,
        y,
        color,
        scale=1.3,
    ):

        cls.draw_centered_text(
            img,
            value,
            center_x,
            y,
            scale,
            color,
            cls.THICKNESS_NORMAL,
            cls.FONT_PRIMARY,
        )
        # ==========================================================
    # Card Title
    # ==========================================================

    @classmethod
    def draw_card_title(
        cls,
        img,
        rect: Rect,
        title,
    ):

        cls.draw_text(
            img,
            title.upper(),
            rect.x + 20,
            rect.y + 28,
            cls.H2,
            cls.ACCENT,
            cls.THICKNESS_NORMAL,
            cls.FONT_PRIMARY,
        )

    # ==========================================================
    # Divider
    # ==========================================================

    @classmethod
    def draw_divider(
        cls,
        img,
        x1,
        y,
        x2,
        color=None,
    ):

        if color is None:
            color = cls.PROGRESS_BG

        cv2.line(
            img,
            (int(x1), int(y)),
            (int(x2), int(y)),
            color,
            1,
            cv2.LINE_AA,
        )

    # ==========================================================
    # Progress Bar
    # ==========================================================

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

        if fill_color is None:
            fill_color = cls.PROGRESS_FILL

        progress = max(0.0, min(1.0, progress))

        cls.draw_rounded_rect(
            img,
            (int(x), int(y)),
            (int(x + width), int(y + height)),
            cls.PROGRESS_BG,
            -1,
            height // 2,
        )

        filled = int(width * progress)

        if filled > 0:

            cls.draw_rounded_rect(
                img,
                (int(x), int(y)),
                (int(x + filled), int(y + height)),
                fill_color,
                -1,
                min(height // 2, max(2, filled // 2)),
            )

    # ==========================================================
    # Chip
    # ==========================================================

    @classmethod
    def draw_chip(
        cls,
        img,
        x,
        y,
        text,
        bg_color,
        text_color=None,
    ):

        if text_color is None:
            text_color = cls.BLACK

        (tw, th), _ = cv2.getTextSize(
            str(text),
            cls.FONT_SECONDARY,
            cls.CAPTION,
            cls.THICKNESS_NORMAL,
        )

        padding_x = 10
        padding_y = 6

        width = tw + (padding_x * 2)
        height = th + (padding_y * 2)

        cls.draw_rounded_rect(
            img,
            (x, y),
            (x + width, y + height),
            bg_color,
            -1,
            cls.RADIUS_BUTTON,
        )

        cls.draw_text(
            img,
            text,
            x + padding_x,
            y + height - padding_y,
            cls.CAPTION,
            text_color,
            cls.THICKNESS_NORMAL,
            cls.FONT_SECONDARY,
        )

    # ==========================================================
    # Status Chip
    # ==========================================================

    @classmethod
    def draw_status_chip(
        cls,
        img,
        x,
        y,
        status,
    ):

        status = str(status).upper()

        if "READY" in status or "COMPLETE" in status:
            color = cls.SUCCESS

        elif "RECORD" in status:
            color = cls.RECORDING

        elif "MOVE" in status:
            color = cls.ACCENT

        elif "COOLDOWN" in status:
            color = cls.ACCENT

        else:
            color = cls.SECONDARY

        cls.draw_chip(
            img,
            int(x),
            int(y),
            status,
            color,
            cls.WHITE,
        )