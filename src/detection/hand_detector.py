"""
SignLanguageAI
Version 0.4.1

Hand Detector
(MediaPipe Tasks - Video Mode)
"""

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:

    def __init__(self, model_path):

        base_options = python.BaseOptions(
            model_asset_path=model_path
        )

        options = vision.HandLandmarkerOptions(

            base_options=base_options,

            running_mode=vision.RunningMode.VIDEO,

            num_hands=2,

            min_hand_detection_confidence=0.5,

            min_hand_presence_confidence=0.5,

            min_tracking_confidence=0.5

        )

        self.detector = vision.HandLandmarker.create_from_options(
            options
        )

    def detect(self, mp_image, timestamp_ms):

        return self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )