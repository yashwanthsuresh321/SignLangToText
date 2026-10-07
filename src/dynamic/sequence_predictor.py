"""
sequence_predictor.py

Version 0.8

Loads a trained LSTM model (to be created later) and performs
prediction on a completed landmark sequence.
"""

from collections import deque
import numpy as np

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


class SequencePredictor:
    def __init__(self,
                 model_path="data/models/dynamic_sign_model.keras",
                 labels=("J", "Z"),
                 sequence_length=20,
                 feature_size=64):

        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.labels = list(labels)

        self.buffer = deque(maxlen=sequence_length)

        self.model = None
        self.model_loaded = False

        if load_model is not None:
            try:
                self.model = load_model(model_path)
                self.model_loaded = True
            except Exception:
                self.model = None
                self.model_loaded = False

    def reset(self):
        self.buffer.clear()

    def add_frame(self, features):
        arr = np.asarray(features, dtype=np.float32).flatten()

        if arr.size != self.feature_size:
            raise ValueError(
                f"Expected {self.feature_size} features, got {arr.size}"
            )

        self.buffer.append(arr)

    def is_ready(self):
        return len(self.buffer) == self.sequence_length

    def progress(self):
        return len(self.buffer) / self.sequence_length

    def predict(self):
        """
        Returns:
            label, confidence

        If the sequence isn't ready or the model isn't loaded,
        returns (None, 0.0).
        """

        if not self.is_ready():
            return None, 0.0

        if not self.model_loaded:
            return None, 0.0

        sequence = np.array(self.buffer, dtype=np.float32)
        sequence = np.expand_dims(sequence, axis=0)

        probs = self.model.predict(sequence, verbose=0)[0]

        idx = int(np.argmax(probs))

        return self.labels[idx], float(probs[idx])
