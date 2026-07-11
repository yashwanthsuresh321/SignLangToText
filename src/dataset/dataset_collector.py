
"""
SignLanguageAI
Version 0.4.1

Professional Dataset Collector
"""

import math
import time
from collections import defaultdict

from src.dataset.csv_writer import CSVWriter
from src.config import Config


class DatasetCollector:

    def __init__(self):
        self.writer = CSVWriter(Config.DATASET_PATH)

        self.current_label = "-"
        self.last_saved = "-"
        self.last_status = "Ready"

        self.total_samples = 0
        self.sample_count = defaultdict(int)
        self.session_count = defaultdict(int)

        self.last_save_time = 0.0
        self.status_timestamp = 0.0

        self.previous_landmarks = None

    # ----------------------------
    # Timing
    # ----------------------------

    def can_save(self):
        return (time.time() - self.last_save_time) >= Config.SAVE_COOLDOWN

    def update_save_timer(self):
        self.last_save_time = time.time()

    # ----------------------------
    # Status
    # ----------------------------

    def set_status(self, message):
        self.last_status = message
        self.status_timestamp = time.time()

    def update_status(self):
        if self.last_status != "Ready":
            if (time.time() - self.status_timestamp) >= 1.0:
                self.last_status = "Ready"

    def update(self):
        self.update_status()

    # ----------------------------
    # Landmark utilities
    # ----------------------------

    def landmark_distance(self, current_landmarks):
        if self.previous_landmarks is None:
            return 9999.0

        total = 0.0

        for cur, prev in zip(current_landmarks, self.previous_landmarks):
            dx = cur.x - prev.x
            dy = cur.y - prev.y
            dz = cur.z - prev.z

            total += math.sqrt(dx * dx + dy * dy + dz * dz)

        return total / len(current_landmarks)

    # ----------------------------
    # Dataset info
    # ----------------------------

    def is_letter_complete(self, label):
        return self.sample_count[label] >= Config.TARGET_SAMPLES

    def remaining_samples(self, label):
        return max(
            Config.TARGET_SAMPLES - self.sample_count[label],
            0
        )

    # ----------------------------
    # Save
    # ----------------------------

    def save_sample(self, label, hand, landmarks):

        self.update_status()

        self.current_label = label

        if self.is_letter_complete(label):
            self.set_status("Letter Complete")
            return False

        if not self.can_save():
            self.set_status("Cooldown")
            return False

        movement = self.landmark_distance(landmarks)

        if movement < Config.MOVEMENT_THRESHOLD:
            self.set_status("Move Hand")
            return False

        self.writer.append(label, hand, landmarks)

        self.previous_landmarks = list(landmarks)

        self.total_samples += 1
        self.sample_count[label] += 1
        self.session_count[label] += 1

        self.last_saved = label
        self.update_save_timer()

        if self.is_letter_complete(label):
            self.set_status(f"{label} Complete")
        else:
            self.set_status("Sample Saved")

        return True

    # ----------------------------
    # Progress
    # ----------------------------

    def get_progress(self, label):
        current = self.sample_count[label]
        target = Config.TARGET_SAMPLES

        progress = min((current / target) * 100.0, 100.0)

        return current, target, progress

    def get_session_count(self, label):
        return self.session_count[label]

    def get_dataset_count(self, label):
        return self.sample_count[label]

    def get_total_samples(self):
        return self.total_samples
