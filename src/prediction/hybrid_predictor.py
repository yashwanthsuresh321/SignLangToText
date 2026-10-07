"""
==========================================================
SignLanguageAI
Version 1.0

Hybrid Predictor

Combines

1. Static MLP Predictor
2. Dynamic LSTM Predictor

The hybrid predictor automatically decides which
prediction should be used.

Author:
==========================================================
"""

from collections import deque

import numpy as np

from src.prediction.predictor import Predictor
from src.dynamic.sequence_predictor import SequencePredictor


class HybridPredictor:

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(
        self,
        model_path,
        scaler_path,
        label_encoder_path,
        dynamic_model_path="data/models/dynamic_sign_model.keras",
        dynamic_threshold=0.70,
        static_threshold=0.75,
        motion_threshold=0.006,
    ):

        self.static_predictor = Predictor(
            model_path=model_path,
            scaler_path=scaler_path,
            label_encoder_path=label_encoder_path,
        )

        self.dynamic_predictor = SequencePredictor(
            model_path=dynamic_model_path
        )

        self.dynamic_threshold = dynamic_threshold
        self.static_threshold = static_threshold

        # Keep the same 20-frame temporal window as the LSTM.
        # This lets us detect movement anywhere in the sequence,
        # rather than checking only the current frame.
        self.motion_history = deque(
            maxlen=self.dynamic_predictor.sequence_length
        )

        # Average frame-to-frame landmark movement required before
        # J/Z is allowed to override the static recognizer.
        self.motion_threshold = motion_threshold

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.static_predictor.reset_history()
        self.dynamic_predictor.reset()
        self.motion_history.clear()

    # ======================================================
    # Dynamic Buffer Progress
    # ======================================================

    def sequence_progress(self):

        return self.dynamic_predictor.progress()

    # ======================================================
    # Ready?
    # ======================================================

    def sequence_ready(self):

        return self.dynamic_predictor.is_ready()

    # ======================================================
    # Sequence Motion
    # ======================================================

    def _calculate_sequence_motion(self):
        """
        Measure movement across the complete 20-frame window.

        We intentionally inspect the whole sequence instead of
        only the latest frame. This is important for J/Z because
        the user can finish the movement and then briefly hold
        the final hand position. A current-frame-only gate would
        incorrectly decide that there was no movement.

        Only the first 63 values are landmark x/y/z coordinates.
        The 64th value is handedness and is excluded.
        """

        if len(self.motion_history) < 2:
            return 0.0

        sequence = np.asarray(
            self.motion_history,
            dtype=np.float32
        )

        landmark_sequence = sequence[:, :63]

        frame_differences = np.diff(
            landmark_sequence,
            axis=0
        )

        # Average absolute landmark displacement per frame.
        motion_score = float(
            np.mean(np.abs(frame_differences))
        )

        return motion_score

    # ======================================================
    # Predict
    # ======================================================

    def predict(
        self,
        landmarks,
        handedness,
    ):

        # ------------------------------------------
        # Static prediction
        # ------------------------------------------

        static_prediction, static_confidence = (

            self.static_predictor.predict_live(

                landmarks,
                handedness,
                threshold=self.static_threshold,

            )

        )

        # ------------------------------------------
        # Raw features for LSTM
        # ------------------------------------------

        raw_features = self.static_predictor.extract_features(

            landmarks,
            handedness,

        )

        # Keep a parallel copy of the exact feature sequence used
        # by the LSTM so the motion gate evaluates the full window.
        self.motion_history.append(
            np.asarray(raw_features, dtype=np.float32)
        )

        self.dynamic_predictor.add_frame(

            raw_features

        )

        # ------------------------------------------
        # If buffer isn't full
        # ------------------------------------------

        if not self.dynamic_predictor.is_ready():

            return {

                "prediction": static_prediction,

                "confidence": static_confidence,

                "source": "STATIC",

                "dynamic_ready": False,

                "dynamic_progress": self.dynamic_predictor.progress(),

                "motion_score": self._calculate_sequence_motion(),

                "motion_detected": False,

            }

        # ------------------------------------------
        # Motion Gate
        # ------------------------------------------
        #
        # The previous implementation checked only the movement
        # between the current frame and the immediately previous
        # frame. At low FPS, or after the user finishes the J/Z
        # movement, that value can be almost zero even though the
        # 20-frame sequence clearly contains a gesture.
        #
        motion_score = self._calculate_sequence_motion()
        motion_detected = motion_score >= self.motion_threshold

        if not motion_detected:

            return {

                "prediction": static_prediction,

                "confidence": static_confidence,

                "source": "STATIC",

                "dynamic_ready": True,

                "dynamic_progress": 1.0,

                "motion_score": motion_score,

                "motion_detected": False,

            }

        # ------------------------------------------
        # Run Dynamic Model
        # ------------------------------------------

        dynamic_prediction, dynamic_confidence = (

            self.dynamic_predictor.predict()

        )

        # ------------------------------------------
        # If LSTM unavailable
        # ------------------------------------------

        if dynamic_prediction is None:

            return {

                "prediction": static_prediction,

                "confidence": static_confidence,

                "source": "STATIC",

                "dynamic_ready": False,

                "dynamic_progress": self.dynamic_predictor.progress(),

                "motion_score": motion_score,

                "motion_detected": True,

            }

        # ------------------------------------------
        # Hybrid Decision
        # ------------------------------------------

        if (

            dynamic_prediction in ("J", "Z")

            and

            dynamic_confidence >= self.dynamic_threshold

        ):

            return {

                "prediction": dynamic_prediction,

                "confidence": dynamic_confidence,

                "source": "DYNAMIC",

                "dynamic_ready": True,

                "dynamic_progress": 1.0,

                "motion_score": motion_score,

                "motion_detected": True,

            }

        # ------------------------------------------
        # Otherwise Static wins
        # ------------------------------------------

        return {

            "prediction": static_prediction,

            "confidence": static_confidence,

            "source": "STATIC",

            "dynamic_ready": True,

            "dynamic_progress": 1.0,

            "motion_score": motion_score,

            "motion_detected": True,

        }

    # ======================================================
    # Convenience Function
    # ======================================================

    def predict_label(
        self,
        landmarks,
        handedness,
    ):

        result = self.predict(

            landmarks,
            handedness,

        )

        return result["prediction"]
    # ======================================================
# Raw Feature Extraction
# ======================================================

    def extract_features(
        self,
        landmarks,
        handedness,
   ):

        return self.static_predictor.extract_features(
            landmarks,
            handedness,
    )