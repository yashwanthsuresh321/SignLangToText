import cv2

from src.ui.layout import Rect
from src.ui.theme import Theme
from src.ui.constants import Constants


class DatasetComponent:
    """
    Dataset Collection Card

    Displays:
    - Current label
    - Collection progress
    - Status
    - Session count
    - Remaining samples
    """

    def draw(self, frame, rect: Rect, data: dict):

        # ---------------------------------------------------------
        # Background
        # ---------------------------------------------------------

        Theme.draw_glass_card(frame, rect)

        current_label = str(data.get("current_label", "-"))
        current_count = int(data.get("current_count", 0))
        target_count = int(data.get("target_count", 300))
        session_count = int(data.get("session_count", 0))
        status = str(data.get("status", "READY"))

        progress = (
            current_count / target_count
            if target_count > 0
            else 0.0
        )

        remaining = max(0, target_count - current_count)

        pad = Constants.PADDING

        left = rect.x + pad
        right = rect.x + rect.w - pad
        center_x = rect.x + rect.w // 2

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------

        Theme.draw_card_title(
            frame,
            rect,
            "Dataset",
        )

        # ---------------------------------------------------------
        # Status Chip
        # ---------------------------------------------------------

        (tw, th), _ = cv2.getTextSize(
            status.upper(),
            Theme.FONT_SECONDARY,
            Theme.CAPTION,
            Theme.THICKNESS_NORMAL,
        )

        chip_width = tw + 20

        chip_x = right - chip_width
        chip_y = rect.y + 12

        Theme.draw_status_chip(
            frame,
            chip_x,
            chip_y,
            status,
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
        # Current Label
        # ---------------------------------------------------------

        Theme.draw_center_value(
            frame,
            current_label,
            center_x,
            rect.y + 72,
            color=Theme.ACCENT,
            scale=1.8,
        )

        # ---------------------------------------------------------
        # Progress Text
        # ---------------------------------------------------------

        Theme.draw_centered_text(
            frame,
            f"{current_count} / {target_count}",
            center_x,
            rect.y + 103,
            scale=0.65,
            color=Theme.WHITE,
            thickness=Theme.THICKNESS_NORMAL,
        )

        # ---------------------------------------------------------
        # Progress Bar
        # ---------------------------------------------------------

        bar_x = left
        bar_y = rect.y + 112
        bar_w = rect.w - (2 * pad)
        bar_h = 10

        Theme.draw_progress_bar(
            frame,
            bar_x,
            bar_y,
            bar_w,
            bar_h,
            progress,
            fill_color=Theme.SUCCESS,
        )

        # ---------------------------------------------------------
        # Bottom Information
        # ---------------------------------------------------------

        bottom_y = rect.y + rect.h - 12

        Theme.draw_text(
            frame,
            f"Session {session_count}",
            left,
            bottom_y,
            scale=Theme.CAPTION,
            color=Theme.SECONDARY,
            thickness=Theme.THICKNESS_THIN,
        )

        remaining_text = f"Left {remaining}"

        (rw, _), _ = cv2.getTextSize(
            remaining_text,
            Theme.FONT_SECONDARY,
            Theme.CAPTION,
            Theme.THICKNESS_THIN,
        )

        Theme.draw_text(
            frame,
            remaining_text,
            right - rw,
            bottom_y,
            scale=Theme.CAPTION,
            color=Theme.SECONDARY,
            thickness=Theme.THICKNESS_THIN,
        )