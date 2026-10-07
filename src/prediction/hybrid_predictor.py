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
        motion_frame_threshold=0.004,
        minimum_motion_frames=5,
        dynamic_confirmation_frames=4,
    ):
        self.dynamic_confirmation_frames = (
            dynamic_confirmation_frames
        )

        self.pending_dynamic_prediction = None
        self.pending_dynamic_count = 0
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
        self.motion_frame_threshold = motion_frame_threshold
        self.minimum_motion_frames = minimum_motion_frames

    # ======================================================
    # Reset
    # ======================================================

    def reset(self):

        self.static_predictor.reset_history()
        self.dynamic_predictor.reset()
        self.motion_history.clear()
        self.pending_dynamic_prediction = None
        self.pending_dynamic_count = 0
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
    def _calculate_sustained_motion(self):
        """
        Determine whether the current sequence contains
        sustained movement.

        This is intentionally different from simply calculating
        average motion.

        A static gesture may contain a short burst of movement
        while the user is forming the hand shape.

        J and Z, however, contain movement across multiple
        consecutive frames.

        Returns
        -------
        tuple
        (
            motion_frame_count,
            maximum_consecutive_motion_frames,
            sustained_motion
        )
        """

        if len(self.motion_history) < 2:
           return 0, 0, False

        sequence = np.asarray(
            self.motion_history,
            dtype=np.float32
        )

        landmark_sequence = sequence[:, :63]

        frame_differences = np.diff(
            landmark_sequence,
            axis=0
        )

        # Calculate average landmark movement for each
        # frame-to-frame transition.
        frame_motion = np.mean(
            np.abs(frame_differences),
            axis=1
        )

        # Frames that contain meaningful movement.
        moving_frames = (
            frame_motion >= self.motion_frame_threshold
        )

        motion_frame_count = int(
            np.sum(moving_frames)
        )

        # Find the longest consecutive run of moving frames.
        maximum_consecutive = 0
        current_consecutive = 0

        for moving in moving_frames:

            if moving:

                current_consecutive += 1

                maximum_consecutive = max(
                    maximum_consecutive,
                    current_consecutive
                )

            else:

                current_consecutive = 0

        sustained_motion = (
            motion_frame_count >= self.minimum_motion_frames
            and
            maximum_consecutive >= 3
        )

        return (
            motion_frame_count,
            maximum_consecutive,
            sustained_motion
        )
    def _confirm_dynamic_prediction(
        self,
        prediction,
    ):
        """
        Require multiple consecutive identical J/Z predictions
        before allowing a dynamic prediction to reach the UI
        and SentenceBuilder.
        """

        if prediction not in ("J", "Z"):

            self.pending_dynamic_prediction = None
            self.pending_dynamic_count = 0

            return False

        # First dynamic prediction.
        if self.pending_dynamic_prediction is None:

            self.pending_dynamic_prediction = prediction
            self.pending_dynamic_count = 1

            return False

        # Same prediction continues.
        if self.pending_dynamic_prediction == prediction:

            self.pending_dynamic_count += 1

        else:

            # Prediction changed from J -> Z or Z -> J.
            self.pending_dynamic_prediction = prediction
            self.pending_dynamic_count = 1

            return False

        return (
            self.pending_dynamic_count
            >= self.dynamic_confirmation_frames
        )
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
        (
            motion_frame_count,
            consecutive_motion_frames,
            sustained_motion,
        ) = self._calculate_sustained_motion()

        if not sustained_motion:
            self.pending_dynamic_prediction = None
            self.pending_dynamic_count = 0

            return {

                "prediction": static_prediction,

                "confidence": static_confidence,

                "source": "STATIC",

                "dynamic_ready": True,

                "dynamic_progress": 1.0,

                "motion_score": motion_score,

                "motion_detected": False,

                "motion_frame_count": motion_frame_count,

                "consecutive_motion_frames": (
                    consecutive_motion_frames
                ),
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
            and
            sustained_motion

        ):
            confirmed = self._confirm_dynamic_prediction(
                dynamic_prediction
            )

            if confirmed:

                return {

                    "prediction": dynamic_prediction,

                    "confidence": dynamic_confidence,

                    "source": "DYNAMIC",

                    "dynamic_ready": True,

                    "dynamic_progress": 1.0,

                    "motion_score": motion_score,

                    "motion_detected": True,

                    "motion_frame_count": motion_frame_count,

                    "consecutive_motion_frames": (
                        consecutive_motion_frames
                    ),

                }

            # J/Z has not been confirmed yet.
            # IMPORTANT:
            # Return the static prediction instead of the
            # temporary J/Z prediction.
            return {

                "prediction": static_prediction,

                "confidence": static_confidence,

                "source": "STATIC",

                "dynamic_ready": True,

                "dynamic_progress": 1.0,

                "motion_score": motion_score,

                "motion_detected": True,

                "motion_frame_count": motion_frame_count,

                "consecutive_motion_frames": (
                    consecutive_motion_frames
                ),

            }
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