"""
SignLanguageAI
Version 0.5.0

Model Trainer
"""

from pathlib import Path

import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


class ModelTrainer:

    def __init__(
        self,
        model,
        model_path
    ):

        self.model = model

        self.model_path = Path(model_path)

        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.history = None

    # ==========================================================
    # Callbacks
    # ==========================================================

    def get_callbacks(self):

        early_stopping = EarlyStopping(

            monitor="val_loss",

            patience=15,

            restore_best_weights=True,

            verbose=1

        )

        checkpoint = ModelCheckpoint(

            filepath=str(self.model_path),

            monitor="val_accuracy",

            save_best_only=True,

            verbose=1

        )

        reduce_lr = ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=1e-6,

            verbose=1

        )

        return [

            early_stopping,

            checkpoint,

            reduce_lr

        ]

    # ==========================================================
    # Train Model
    # ==========================================================

    def train(

        self,

        X_train,

        y_train,

        X_validation,

        y_validation,

        epochs=100,

        batch_size=16

    ):

        print()

        print("=" * 60)

        print("           TRAINING STARTED")

        print("=" * 60)

        self.history = self.model.fit(

            X_train,

            y_train,

            validation_data=(

                X_validation,

                y_validation

            ),

            epochs=epochs,

            batch_size=batch_size,

            callbacks=self.get_callbacks(),

            verbose=1

        )

        print()

        print("=" * 60)

        print("         TRAINING COMPLETED")

        print("=" * 60)

        return self.history

    # ==========================================================
    # Save Model
    # ==========================================================

    def save(self):

        self.model.save(

            self.model_path

        )

        print(

            f"\nModel Saved : {self.model_path}"

        )

    # ==========================================================
    # History
    # ==========================================================

    def get_history(self):

        return self.history

    # ==========================================================
    # Training Summary
    # ==========================================================

    def summary(self):

        if self.history is None:

            return

        final_train_accuracy = (

            self.history.history["accuracy"][-1]

        )

        final_validation_accuracy = (

            self.history.history["val_accuracy"][-1]

        )

        final_train_loss = (

            self.history.history["loss"][-1]

        )

        final_validation_loss = (

            self.history.history["val_loss"][-1]

        )

        print()

        print("=" * 60)

        print("          TRAINING SUMMARY")

        print("=" * 60)

        print(

            f"Training Accuracy   : "

            f"{final_train_accuracy:.4f}"

        )

        print(

            f"Validation Accuracy : "

            f"{final_validation_accuracy:.4f}"

        )

        print(

            f"Training Loss       : "

            f"{final_train_loss:.4f}"

        )

        print(

            f"Validation Loss     : "

            f"{final_validation_loss:.4f}"

        )

        print("=" * 60)