"""
sequence_dataset.py

Version 0.8.1

Stores, validates, saves and loads dynamic gesture sequences
for J and Z recognition.

Features
--------
✓ Stores dynamic gesture sequences
✓ Validates sequence dimensions
✓ Automatic directory creation
✓ Safe loading
✓ Corruption handling
✓ Compressed dataset storage
✓ Resume collection across sessions

Dataset format
--------------
X : (N, sequence_length, feature_size)
y : (N,)
"""

import os
import numpy as np


class SequenceDataset:
    def __init__(
        self,
        sequence_length=20,
        feature_size=64,
    ):

        self.sequence_length = sequence_length
        self.feature_size = feature_size

        self.X = []
        self.y = []

        self.label_map = {
            "J": 0,
            "Z": 1
        }

        self.inverse_label_map = {
            0: "J",
            1: "Z"
        }

    # ==========================================================
    # Add Sequence
    # ==========================================================

    def add_sequence(self, sequence, label):

        sequence = np.asarray(sequence, dtype=np.float32)

        expected_shape = (
            self.sequence_length,
            self.feature_size
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Expected shape {expected_shape}, "
                f"got {sequence.shape}"
            )

        label = label.upper()

        if label not in self.label_map:
            raise ValueError(
                "Only J and Z are supported."
            )

        self.X.append(sequence)
        self.y.append(self.label_map[label])

    # ==========================================================
    # Information
    # ==========================================================

    def num_sequences(self):
        return len(self.X)

    def label_counts(self):

        counts = {
            "J": 0,
            "Z": 0
        }

        for label in self.y:
            counts[self.inverse_label_map[int(label)]] += 1

        return counts

    def clear(self):

        self.X.clear()
        self.y.clear()

    # ==========================================================
    # Data Access
    # ==========================================================

    def get_data(self):

        if len(self.X) == 0:

            return (
                np.empty(
                    (
                        0,
                        self.sequence_length,
                        self.feature_size
                    ),
                    dtype=np.float32
                ),
                np.empty(
                    (0,),
                    dtype=np.int32
                )
            )

        X = np.stack(self.X).astype(np.float32)
        y = np.asarray(
            self.y,
            dtype=np.int32
        )

        return X, y

    # ==========================================================
    # File Utilities
    # ==========================================================

    def exists(self, filepath):

        return os.path.exists(filepath)

    # ==========================================================
    # Save Dataset
    # ==========================================================

    def save(self, filepath):

        directory = os.path.dirname(filepath)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        X, y = self.get_data()

        np.savez_compressed(
            filepath,
            X=X,
            y=y
        )

        return True

    # ==========================================================
    # Load Dataset
    # ==========================================================

    def load(self, filepath):

        if not os.path.exists(filepath):
            return False

        try:

            data = np.load(
                filepath,
                allow_pickle=False
            )

            self.X = [
                np.asarray(x, dtype=np.float32)
                for x in data["X"]
            ]

            self.y = [
                int(v)
                for v in data["y"]
            ]

            return True

        except Exception as e:

            print(
                "[Dynamic Dataset] Failed to load:"
            )
            print(e)

            self.clear()

            return False

    # ==========================================================
    # Summary
    # ==========================================================

    def summary(self):

        counts = self.label_counts()

        return {

            "total_sequences":
                self.num_sequences(),

            "J":
                counts["J"],

            "Z":
                counts["Z"],

            "sequence_length":
                self.sequence_length,

            "feature_size":
                self.feature_size
        }

    # ==========================================================
    # Pretty Print
    # ==========================================================

    def print_summary(self):

        summary = self.summary()

        print("\n" + "=" * 55)
        print("        DYNAMIC DATASET SUMMARY")
        print("=" * 55)
        print(f"Total Sequences : {summary['total_sequences']}")
        print(f"J Sequences     : {summary['J']}")
        print(f"Z Sequences     : {summary['Z']}")
        print(f"Sequence Length : {summary['sequence_length']}")
        print(f"Feature Size    : {summary['feature_size']}")
        print("=" * 55)