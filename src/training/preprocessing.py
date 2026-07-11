"""
SignLanguageAI
Version 0.5.0

Data Preprocessing
"""

from pathlib import Path

import joblib

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split
)


class DataPreprocessor:

    def __init__(self):

        self.label_encoder = LabelEncoder()

        self.scaler = StandardScaler()

    # ==========================================================
    # Label Encoding
    # ==========================================================

    def encode_labels(
        self,
        labels
    ):

        encoded = self.label_encoder.fit_transform(
            labels
        )

        return encoded

    # ==========================================================
    # Train / Test Split
    # ==========================================================

    def split_data(
        self,
        X,
        y,
        test_size=0.20,
        random_state=42
    ):

        return train_test_split(

            X,

            y,

            test_size=test_size,

            random_state=random_state,

            stratify=y,

            shuffle=True

        )

    # ==========================================================
    # Feature Scaling
    # ==========================================================

    def scale_features(
        self,
        X_train,
        X_test
    ):

        X_train = self.scaler.fit_transform(
            X_train
        )

        X_test = self.scaler.transform(
            X_test
        )

        return X_train, X_test

    # ==========================================================
    # Save Label Encoder
    # ==========================================================

    def save_label_encoder(
        self,
        path
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(

            self.label_encoder,

            path

        )

        print(
            f"Label Encoder Saved : {path}"
        )

    # ==========================================================
    # Save Feature Scaler
    # ==========================================================

    def save_scaler(
        self,
        path
    ):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(

            self.scaler,

            path

        )

        print(
            f"Feature Scaler Saved : {path}"
        )

    # ==========================================================
    # Complete Pipeline
    # ==========================================================

    def prepare(
        self,
        X,
        y
    ):

        print("\nEncoding Labels...")

        y = self.encode_labels(
            y
        )

        print("Splitting Dataset...")

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = self.split_data(

            X,

            y

        )

        print("Scaling Features...")

        (
            X_train,
            X_test
        ) = self.scale_features(

            X_train,

            X_test

        )

        print()

        print(
            f"Training Samples : {len(X_train)}"
        )

        print(
            f"Testing Samples  : {len(X_test)}"
        )

        print(
            f"Input Features   : {X_train.shape[1]}"
        )

        print()

        return (

            X_train,

            X_test,

            y_train,

            y_test

        )