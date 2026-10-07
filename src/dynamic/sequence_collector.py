"""
sequence_collector.py

Dynamic sequence collector for J and Z gestures.
"""

from collections import deque
import numpy as np


class SequenceCollector:
    def __init__(self, sequence_length=20, feature_size=64):
        self.sequence_length = sequence_length
        self.feature_size = feature_size

        self._buffer = deque(maxlen=sequence_length)

        self.current_label = None
        self.recording = False
        self.sequence_count = 0

    # ==========================================================
    # Start Recording
    # ==========================================================

    def start(self, label):
        """
        Start recording a new dynamic gesture sequence.
        """

        if self.recording:
            raise RuntimeError(
                "A recording is already in progress."
            )

        label = label.upper()

        if label not in ("J", "Z"):
            raise ValueError(
                "Only J and Z are supported."
            )

        self.current_label = label
        self.recording = True
        self._buffer.clear()

    # ==========================================================
    # Cancel Recording
    # ==========================================================

    def cancel(self):
        """
        Cancel the current recording.
        """

        self.recording = False
        self.current_label = None
        self._buffer.clear()

    # ==========================================================
    # Add Frame
    # ==========================================================

    def update(self, features):
        """
        Add one feature vector to the sequence.
        """

        # Ignore frames unless actively recording
        if not self.recording:
            return

        arr = np.asarray(
            features,
            dtype=np.float32
        ).flatten()

        if arr.size != self.feature_size:
            raise ValueError(
                f"Expected {self.feature_size} features, got {arr.size}"
            )

        self._buffer.append(arr)

    # ==========================================================
    # Progress
    # ==========================================================

    def progress(self):
        """
        Returns recording progress between 0.0 and 1.0.
        """

        return len(self._buffer) / self.sequence_length

    # ==========================================================
    # Frames Collected
    # ==========================================================

    def frames_collected(self):
        return len(self._buffer)

    # ==========================================================
    # Frames Remaining
    # ==========================================================

    def frames_remaining(self):
        return self.sequence_length - len(self._buffer)

    # ==========================================================
    # Current Sequence Length
    # ==========================================================

    @property
    def sequence(self):
        """
        Current number of frames collected.
        """

        return len(self._buffer)

    # ==========================================================
    # Completion Check
    # ==========================================================

    def is_complete(self):
        return len(self._buffer) == self.sequence_length

    # ==========================================================
    # Retrieve Completed Sequence
    # ==========================================================

    def get_sequence(self):
        """
        Returns (sequence, label) when recording is complete.
        Otherwise returns (None, None).
        """

        if not self.is_complete():
            return None, None

        sequence = np.stack(self._buffer)

        label = self.current_label

        self.sequence_count += 1

        self.cancel()

        return sequence, label

    # ==========================================================
    # Status
    # ==========================================================

    @property
    def status(self):
        if not self.recording:
            return "IDLE"

        if self.is_complete():
            return "COMPLETE"

        return "RECORDING"