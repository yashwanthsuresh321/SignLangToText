import cv2

from src.ui.layout import Rect
from src.ui.theme import Theme
from src.ui.constants import Constants


class DynamicPanelComponent:
    """
    Dynamic Dataset Panel

    Displays:
    - Recording status
    - Current dynamic sign
    - Frame progress
    - Dataset progress
    - J/Z distribution
    """

    def draw(self, frame, rect: Rect, data: dict):

        # ---------------------------------------------------------
        # Background
        # ---------------------------------------------------------

        Theme.draw_glass_card(frame, rect)

        dynamic_label = str(data.get("label", "-"))
        dynamic_frames = int(data.get("frames", 0))
        dynamic_target = int(data.get("target", 20))

        dynamic_recording = bool(data.get("recording", False))

        dynamic_collected = int(data.get("collected", 0))
        dynamic_goal = int(data.get("goal", 150))

        dynamic_j_count = int(data.get("j_count", 0))
        dynamic_z_count = int(data.get("z_count", 0))

        frame_progress = (
            dynamic_frames / dynamic_target
            if dynamic_target > 0
            else 0.0
        )

        dataset_progress = (
            dynamic_collected / dynamic_goal
            if dynamic_goal > 0
            else 0.0
        )

        remaining = max(0, dynamic_goal - dynamic_collected)

        pad = Constants.PADDING

        left = rect.x + pad
        right = rect.x + rect.w - pad

        # ---------------------------------------------------------
        # Header
        # ---------------------------------------------------------

        Theme.draw_card_title(
            frame,
            rect,
            "Dynamic Dataset",
        )

        status_text = "RECORDING" if dynamic_recording else "STANDBY"

        (tw, _), _ = cv2.getTextSize(
            status_text,
            Theme.FONT_SECONDARY,
            Theme.CAPTION,
            Theme.THICKNESS_NORMAL,
        )

        chip_width = tw + 20

        Theme.draw_status_chip(
            frame,
            right - chip_width,
            rect.y + 12,
            status_text,
        )

        Theme.draw_divider(
            frame,
            left,
            rect.y + 42,
            right,
        )

        # ---------------------------------------------------------
        # Layout
        # ---------------------------------------------------------

        usable_width = rect.w - (2 * pad)

        col_width = usable_width // 3

        col1_x = left
        col2_x = left + col_width
        col3_x = left + (2 * col_width)

        center1 = col1_x + (col_width // 2)
        center2 = col2_x + (col_width // 2)
        center3 = col3_x + (col_width // 2)

        top_y = rect.y + 60

        # =========================================================
        # COLUMN 1
        # Recording
        # =========================================================

        Theme.draw_centered_text(
            frame,
            "Recording",
            center1,
            top_y,
            scale=Theme.CAPTION,
            color=Theme.SECONDARY,
        )

        Theme.draw_recording_dot(
            frame,
            center1 - 34,
            top_y + 22,
            radius=5,
            is_recording=dynamic_recording,
        )

        Theme.draw_center_value(
            frame,
            dynamic_label,
            center1,
            top_y + 42,
            color=Theme.RECORDING,
            scale=1.4,
        )

        Theme.draw_progress_bar(
            frame,
            col1_x,
            top_y + 58,
            col_width - 18,
            Constants.DYNAMIC_PROGRESS_BAR_HEIGHT,
            frame_progress,
            fill_color=Theme.RECORDING,
        )

        Theme.draw_centered_text(
            frame,
            f"{dynamic_frames}/{dynamic_target} Frames",
            center1,
            top_y + 78,
            scale=Theme.CAPTION,
            color=Theme.SECONDARY,
        )

        # =========================================================
        # COLUMN 2
        # Dataset
        # =========================================================

        Theme.draw_centered_text(
            frame,
            "Dataset",
            center2,
            top_y,
            scale=Theme.CAPTION,
            color=Theme.SECONDARY,
        )

        Theme.draw_center_value(
            frame,
            str(dynamic_collected),
            center2,
            top_y + 42,
            color=Theme.SUCCESS,
            scale=1.4,
        )

        Theme.draw_progress_bar(
            frame,
            col2_x,
            top_y + 58,
            col_width - 18,
            Constants.DYNAMIC_PROGRESS_BAR_HEIGHT,
            dataset_progress,
            fill_color=Theme.SUCCESS,
        )

        Theme.draw_centered_text(
            frame,
            f"{remaining} Remaining",
            center2,
            top_y + 78,
            scale=Theme.CAPTION,
            color=Theme.SECONDARY,
        )

        # =========================================================
        # COLUMN 3
        # Distribution
        # =========================================================

        Theme.draw_centered_text(
            frame,
            "Distribution",
            center3,
            top_y,
            scale=Theme.CAPTION,
            color=Theme.SECONDARY,
        )
                # ---------------------------------------------------------
        # J Statistics
        # ---------------------------------------------------------

        Theme.draw_centered_text(
            frame,
            "J",
            center3 - 35,
            top_y + 30,
            scale=Theme.BODY,
            color=Theme.ACCENT,
            thickness=Theme.THICKNESS_NORMAL,
        )

        Theme.draw_center_value(
            frame,
            str(dynamic_j_count),
            center3 - 35,
            top_y + 55,
            color=Theme.WHITE,
            scale=1.0,
        )

        # ---------------------------------------------------------
        # Z Statistics
        # ---------------------------------------------------------

        Theme.draw_centered_text(
            frame,
            "Z",
            center3 + 35,
            top_y + 30,
            scale=Theme.BODY,
            color=Theme.ACCENT,
            thickness=Theme.THICKNESS_NORMAL,
        )

        Theme.draw_center_value(
            frame,
            str(dynamic_z_count),
            center3 + 35,
            top_y + 55,
            color=Theme.WHITE,
            scale=1.0,
        )

        # ---------------------------------------------------------
        # Bottom Divider
        # ---------------------------------------------------------

        bottom_divider_y = rect.y + rect.h - 32

        Theme.draw_divider(
            frame,
            left,
            bottom_divider_y,
            right,
        )

        # ---------------------------------------------------------
        # Bottom Summary
        # ---------------------------------------------------------

        bottom_y = rect.y + rect.h - 12

        if dynamic_recording:
            summary = (
                f"Recording {dynamic_label} "
                f"({dynamic_frames}/{dynamic_target})"
            )
            summary_color = Theme.RECORDING
        else:
            summary = "Waiting to record J or Z"
            summary_color = Theme.SECONDARY

        Theme.draw_text(
            frame,
            summary,
            left,
            bottom_y,
            scale=Theme.CAPTION,
            color=summary_color,
            thickness=Theme.THICKNESS_THIN,
        )

        completion = (
            f"{int(dataset_progress * 100)}%"
        )

        (tw, _), _ = cv2.getTextSize(
            completion,
            Theme.FONT_SECONDARY,
            Theme.CAPTION,
            Theme.THICKNESS_THIN,
        )

        Theme.draw_text(
            frame,
            completion,
            right - tw,
            bottom_y,
            scale=Theme.CAPTION,
            color=Theme.SUCCESS,
            thickness=Theme.THICKNESS_THIN,
        )