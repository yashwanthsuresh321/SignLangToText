import cv2

from src.ui.layout import Rect
from src.ui.theme import Theme
from src.ui.constants import Constants


class PredictionComponent:
    """
    Modern Prediction Card

    Displays:
    - Current prediction
    - Confidence
    - Prediction source
    - Hold progress
    """

    def draw(self, frame, rect: Rect, data: dict):

        # ---------------------------------------------------------
        # Background
        # ---------------------------------------------------------

        Theme.draw_glass_card(frame, rect)

        prediction = str(data.get("prediction", "-"))
        confidence = float(data.get("confidence", 0.0))
        hold_progress = max(0.0, min(1.0, float(data.get("hold_progress", 0.0))))
        source = data.get("source", "STATIC")

        pad = Constants.PADDING

        left = rect.x + pad
        top = rect.y + pad

        center_x = rect.x + rect.w // 2

        # ---------------------------------------------------------
        # Choose prediction color
        # ---------------------------------------------------------

        if prediction == "-":
            pred_color = Theme.SECONDARY
        elif confidence >= 95:
            pred_color = Theme.SUCCESS
        elif confidence >= 80:
            pred_color = Theme.ACCENT
        else:
            pred_color = Theme.ERROR

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------

        cv2.putText(
            frame,
            "PREDICTION",
            (left, top + 10),
            Theme.FONT_PRIMARY,
            Theme.H2,
            Theme.ACCENT,
            Theme.THICKNESS_NORMAL,
            cv2.LINE_AA,
        )

        # ---------------------------------------------------------
        # Large Prediction Letter
        # ---------------------------------------------------------

        (text_w, text_h), _ = cv2.getTextSize(
            prediction,
            Theme.FONT_PRIMARY,
            2.2,
            Theme.THICKNESS_THICK,
        )

        pred_x = center_x - text_w // 2
        pred_y = rect.y + 72

        cv2.putText(
            frame,
            prediction,
            (pred_x, pred_y),
            Theme.FONT_PRIMARY,
            2.2,
            pred_color,
            Theme.THICKNESS_THICK,
            cv2.LINE_AA,
        )

        # ---------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------

        confidence_text = f"{confidence:.1f}%"

        (conf_w, _), _ = cv2.getTextSize(
            confidence_text,
            Theme.FONT_SECONDARY,
            0.75,
            Theme.THICKNESS_NORMAL,
        )

        conf_x = center_x - conf_w // 2
        conf_y = pred_y + 28

        cv2.putText(
            frame,
            confidence_text,
            (conf_x, conf_y),
            Theme.FONT_SECONDARY,
            0.75,
            Theme.WHITE,
            Theme.THICKNESS_NORMAL,
            cv2.LINE_AA,
        )

        # ---------------------------------------------------------
        # Source Badge
        # ---------------------------------------------------------

        badge_color = (
            Theme.SUCCESS
            if source.upper() == "DYNAMIC"
            else Theme.ACCENT
        )

        badge_text = source.upper()

        (badge_w, badge_h), _ = cv2.getTextSize(
            badge_text,
            Theme.FONT_SECONDARY,
            0.45,
            Theme.THICKNESS_NORMAL,
        )

        badge_padding_x = 10
        badge_padding_y = 5

        badge_total_w = badge_w + badge_padding_x * 2
        badge_total_h = badge_h + badge_padding_y * 2

        badge_x = center_x - badge_total_w // 2
        badge_y = conf_y + 12

        Theme.draw_rounded_rect(
            frame,
            (badge_x, badge_y),
            (
                badge_x + badge_total_w,
                badge_y + badge_total_h,
            ),
            badge_color,
            -1,
            8,
        )

        cv2.putText(
            frame,
            badge_text,
            (
                badge_x + badge_padding_x,
                badge_y + badge_total_h - badge_padding_y - 2,
            ),
            Theme.FONT_SECONDARY,
            0.45,
            Theme.BLACK,
            Theme.THICKNESS_NORMAL,
            cv2.LINE_AA,
        )

        # ---------------------------------------------------------
        # Hold Label
        # ---------------------------------------------------------

        hold_label_y = rect.y + rect.h - 34

        cv2.putText(
            frame,
            "HOLD",
            (left, hold_label_y),
            Theme.FONT_SECONDARY,
            Theme.CAPTION,
            Theme.SECONDARY,
            Theme.THICKNESS_THIN,
            cv2.LINE_AA,
        )

        # ---------------------------------------------------------
        # Hold Progress Bar
        # ---------------------------------------------------------

        bar_x = left
        bar_y = hold_label_y + 8

        bar_w = rect.w - (2 * pad)
        bar_h = 10

        Theme.draw_rounded_rect(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h),
            Theme.PROGRESS_BG,
            -1,
            bar_h // 2,
        )

        filled = int(bar_w * hold_progress)

        if filled > 0:

            Theme.draw_rounded_rect(
                frame,
                (bar_x, bar_y),
                (bar_x + filled, bar_y + bar_h),
                Theme.SUCCESS,
                -1,
                min(bar_h // 2, max(1, filled // 2)),
            )

        # ---------------------------------------------------------
        # Hold Percentage
        # ---------------------------------------------------------

        percent_text = f"{int(hold_progress * 100)}%"

        (pw, _), _ = cv2.getTextSize(
            percent_text,
            Theme.FONT_SECONDARY,
            0.45,
            Theme.THICKNESS_NORMAL,
        )

        cv2.putText(
            frame,
            percent_text,
            (
                bar_x + bar_w - pw,
                hold_label_y,
            ),
            Theme.FONT_SECONDARY,
            0.45,
            Theme.WHITE,
            Theme.THICKNESS_NORMAL,
            cv2.LINE_AA,
        )