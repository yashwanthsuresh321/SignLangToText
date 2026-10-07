"""
SignLanguageAI
Version 0.6.0

Live Prediction Module
"""

from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from collections import deque, Counter

class Predictor:

    def __init__(
        self,
        model_path,
        scaler_path,
        label_encoder_path
    ):

        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.label_encoder_path = Path(label_encoder_path)

        self.model = None
        self.scaler = None
        self.label_encoder = None
        # ==========================================================
        # Live Prediction Buffers
        # ==========================================================

        self.window_size = 3

        self.prediction_buffer = deque(
            maxlen=self.window_size
    )

        self.confidence_buffer = deque(
            maxlen=self.window_size
    )
        self.load()

    # ==========================================================
    # Load Resources
    # ==========================================================

    def load(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model not found:\n{self.model_path}"
            )

        if not self.scaler_path.exists():

            raise FileNotFoundError(
                f"Scaler not found:\n{self.scaler_path}"
            )

        if not self.label_encoder_path.exists():

            raise FileNotFoundError(
                f"Label Encoder not found:\n{self.label_encoder_path}"
            )

        print("\nLoading AI Model...")

        self.model = tf.keras.models.load_model(
            self.model_path
        )

        self.scaler = joblib.load(
            self.scaler_path
        )

        self.label_encoder = joblib.load(
            self.label_encoder_path
        )

        print("Model Loaded Successfully")
    # ==========================================================
    # Reset Prediction History
    # ==========================================================

    def reset_history(self):

        self.prediction_buffer.clear()

        self.confidence_buffer.clear()


    # ==========================================================
    # Add New Prediction
    # ==========================================================

    def add_prediction(

        self,

        prediction,

        confidence
    ):

        self.prediction_buffer.append(

            prediction
        )
        self.confidence_buffer.append(

            confidence

    )

    # ==========================================================
    # Stable Prediction
    # ==========================================================

    def get_stable_prediction(self):

        if len(

            self.prediction_buffer

        ) == 0:

            return "-", 0.0

        prediction_counter = Counter(

            self.prediction_buffer

        )

        prediction, count = (

            prediction_counter.most_common(1)[0]

        )

        # ------------------------------------------
        # Require Majority
        # ------------------------------------------

        majority = (

            self.window_size // 2

        ) + 1

        if count < majority:

            return "-", 0.0

        avg_confidence = (

            sum(self.confidence_buffer)

            / len(self.confidence_buffer)

        )

        return prediction, avg_confidence
    # ==========================================================
    # Live Prediction
    # ==========================================================

    def predict_live(
        self,
        landmarks,
        handedness,
        threshold=0.75
    ):

        # -----------------------------
        # Raw Prediction
        # -----------------------------

        prediction, confidence = self.predict(
            landmarks,
            handedness
        )

        # -----------------------------
        # Reject Low Confidence
        # -----------------------------

        if confidence < threshold:

            return self.get_stable_prediction()

        # -----------------------------
        # Store Prediction
        # -----------------------------

        self.add_prediction(
            prediction,
            confidence
        )

        # -----------------------------
        # Stable Prediction
        # -----------------------------

        stable_prediction, stable_confidence = (

            self.get_stable_prediction()

        )

        return (

            stable_prediction,

            stable_confidence

        )
    # ==========================================================
    # Landmark → Feature Vector
    # ==========================================================

    def prepare_features(
        self,
        landmarks,
        handedness
    ):

        features = []

        for landmark in landmarks:

            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        # Hand feature
        if handedness == "Left":

            features.append(0.0)

        else:

            features.append(1.0)

        features = np.array(
            features,
            dtype=np.float32
        ).reshape(1, -1)

        features = self.scaler.transform(
            features
        )

        return features


    # ==========================================================
    # Raw Landmark → Feature Vector (Unscaled)
    # Used by the Dynamic LSTM sequence collector
    # ==========================================================

    def extract_features(
        self,
        landmarks,
        handedness
    ):
        """
        Returns the raw 64-dimensional feature vector
        without applying the StandardScaler.
        """
        if landmarks is None:
            raise ValueError("extract_features() received None landmarks")
        features = []

        for landmark in landmarks:
            features.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

        if handedness == "Left":
            features.append(0.0)
        else:
            features.append(1.0)

        return np.asarray(features, dtype=np.float32)

    # ==========================================================
    # Predict
    # ==========================================================

    def predict(
        self,
        landmarks,
        handedness
    ):

        raw_features = self.extract_features(
            landmarks,
            handedness
        )

        features = self.scaler.transform(raw_features.reshape(1, -1))

        # Legacy path removed
        # features = self.prepare_features(
        #     landmarks,
        #     handedness
        # )

        probabilities = self.model.predict(
            features,
            verbose=0
        )

        class_index = np.argmax(
            probabilities
        )

        confidence = float(
            probabilities[0][class_index]
        )

        prediction = self.label_encoder.inverse_transform(
            [class_index]
        )[0]

        return prediction, confidence

    # ==========================================================
    # Predict With Threshold
    # ==========================================================

    def predict_with_threshold(
        self,
        landmarks,
        handedness,
        threshold=0.75
    ):

        prediction, confidence = self.predict(
            landmarks,
            handedness
        )

        if confidence < threshold:

            return "-", confidence

        return prediction, confidence