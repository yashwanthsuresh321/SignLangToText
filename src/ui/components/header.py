import cv2

from src.ui.layout import Rect
from src.ui.theme import Theme


class HeaderComponent:
    """
    Modern dashboard header.

    Displays:
    - FPS
    - Handedness
    - Number of hands
    - Detection confidence
    - Camera resolution
    """

    def draw(self, frame, rect: Rect, data: dict):

        Theme.draw_glass_card(frame, rect)

        metrics = [
            {
                "text": f"{data.get('fps', 0)} FPS",
                "color": Theme.SUCCESS,
            },
            {
                "text": str(data.get("handedness", "-")).upper(),
                "color": Theme.ACCENT,
            },
            {
                "text": (
                f"{data.get('hand_count', 0)} HAND"
                if data.get("hand_count", 0) == 1
                else f"{data.get('hand_count', 0)} HANDS"
                ),
                "color": Theme.WHITE,
            },
            {
                "text": f"{data.get('confidence', 0.0):.1f}%",
                "color": Theme.SUCCESS,
            },
            {
                "text": str(data.get("resolution", "")),
                "color": Theme.SECONDARY,
            },
        ]

        count = len(metrics)

        section_width = rect.w / count

        chip_y = rect.y + (rect.h // 2) - 13

        for i, metric in enumerate(metrics):

            text = metric["text"]
            color = metric["color"]

            (tw, th), _ = cv2.getTextSize(
                text,
                Theme.FONT_SECONDARY,
                Theme.CAPTION,
                Theme.THICKNESS_NORMAL,
            )

            chip_width = tw + 20

            x = int(
                rect.x
                + (i * section_width)
                + (section_width - chip_width) / 2
            )

            Theme.draw_chip(
                frame,
                x,
                chip_y,
                text,
                bg_color=color,
            )