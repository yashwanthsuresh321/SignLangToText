"""
SignLanguageAI
Version 0.5.0

Dataset Loader
"""

from pathlib import Path

import pandas as pd
import numpy as np


class DatasetLoader:

    def __init__(self, csv_path):

        self.csv_path = Path(csv_path)

        self.dataframe = None

    # ==========================================================
    # Load CSV
    # ==========================================================

    def load(self):

        if not self.csv_path.exists():

            raise FileNotFoundError(

                f"\nDataset not found:\n{self.csv_path}"

            )

        self.dataframe = pd.read_csv(

            self.csv_path

        )

        print("\nDataset Loaded Successfully")

        print(f"Samples : {len(self.dataframe)}")

        print(f"Columns : {len(self.dataframe.columns)}")

        return self.dataframe

    # ==========================================================
    # Validate Dataset
    # ==========================================================

    def validate(self):

        if self.dataframe is None:

            raise RuntimeError(

                "Dataset not loaded."

            )

        required_columns = [

            "label",

            "hand"

        ]

        for column in required_columns:

            if column not in self.dataframe.columns:

                raise ValueError(

                    f"Missing column: {column}"

                )

        if self.dataframe.isnull().values.any():

            raise ValueError(

                "Dataset contains missing values."

            )

        print("Dataset Validation Passed")

    # ==========================================================
    # Convert Hand to Numeric
    # ==========================================================

    def encode_hand(self):

        hand = self.dataframe["hand"].map({

            "Left": 0,

            "Right": 1

        })

        return hand.astype(np.float32)

    # ==========================================================
    # Extract Features
    # ==========================================================

    def get_features(self):

        landmark_columns = [

            column

            for column in self.dataframe.columns

            if column.startswith(

                ("x", "y", "z")

            )

        ]

        landmarks = self.dataframe[

            landmark_columns

        ].values.astype(

            np.float32

        )

        hand = self.encode_hand().values.reshape(

            -1,

            1

        )

        features = np.concatenate(

            (

                landmarks,

                hand

            ),

            axis=1

        )

        return features

    # ==========================================================
    # Labels
    # ==========================================================

    def get_labels(self):

        return self.dataframe[

            "label"

        ].values

    # ==========================================================
    # Dataset Summary
    # ==========================================================

    def summary(self):

        print("\n========== DATASET SUMMARY ==========")

        print(

            self.dataframe["label"]

            .value_counts()

            .sort_index()

        )

        print("\nHand Distribution")

        print(

            self.dataframe["hand"]

            .value_counts()

        )

        print("=====================================\n")

    # ==========================================================
    # Complete Loader
    # ==========================================================

    def prepare(self):

        self.load()

        self.validate()

        self.summary()

        X = self.get_features()

        y = self.get_labels()

        print(

            f"Feature Shape : {X.shape}"

        )

        print(

            f"Label Shape   : {y.shape}"

        )

        return X, y