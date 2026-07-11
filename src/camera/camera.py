"""
SignLanguageAI
Phase 2.3A

Camera Module
"""

import cv2

from src.config import Config


class Camera:

    def __init__(self):

        self.cap = cv2.VideoCapture(
            Config.CAMERA_INDEX
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            Config.FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            Config.FRAME_HEIGHT
        )

    def read(self):

        success, frame = self.cap.read()

        if not success:
            return None

        if Config.MIRROR_CAMERA:
            frame = cv2.flip(frame, 1)

        return frame

    def get_resolution(self):

        width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        return f"{width} x {height}"

    def is_opened(self):

        return self.cap.isOpened()

    def release(self):

        self.cap.release()

    def create_window(self):

        cv2.namedWindow(
            Config.WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )