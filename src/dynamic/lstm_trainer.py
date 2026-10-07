"""
SignLanguageAI
Version 2.0

Professional Dynamic Gesture Trainer

Optimized for:
- J (Dynamic)
- Z (Dynamic)

Input:
    X : (N, 20, 64)
    y : (N,)
"""

from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

# ==========================================================
# Paths
# ==========================================================

ROOT = Path("data")

DATASET_PATH = ROOT / "datasets" / "dynamic_dataset.npz"

MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "dynamic_sign_model.keras"

HISTORY_PATH = MODEL_DIR / "dynamic_training_history.pkl"

LABELS = {
    0: "J",
    1: "Z",
}

# ==========================================================
# Trainer
# ==========================================================


class LSTMTrainer:

    def __init__(
        self,
        sequence_length=20,
        feature_size=64,
        num_classes=2,
    ):

        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.num_classes = num_classes

        self.model = None
        self.history = None

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ======================================================
    # Load Dataset
    # ======================================================

    def load_dataset(self):

        print()
        print("=" * 60)
        print("Loading Dynamic Dataset")
        print("=" * 60)

        if not DATASET_PATH.exists():

            raise FileNotFoundError(
                f"\nDataset not found:\n{DATASET_PATH}"
            )

        data = np.load(
            DATASET_PATH,
            allow_pickle=False,
        )

        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)

        print(f"Sequences       : {len(X)}")
        print(f"Sequence Length : {X.shape[1]}")
        print(f"Features        : {X.shape[2]}")
        print(f"Classes         : {len(np.unique(y))}")
        print()

        unique, counts = np.unique(
            y,
            return_counts=True,
        )

        for cls, count in zip(unique, counts):

            print(f"{LABELS[int(cls)]} : {count}")

        print("=" * 60)

        return X, y

    # ======================================================
    # Build Model
    # ======================================================

    def build_model(self):

        print()
        print("=" * 60)
        print("Building Optimized LSTM Model")
        print("=" * 60)

        model = Sequential(

            [

                Input(
                    shape=(
                        self.sequence_length,
                        self.feature_size,
                    )
                ),

                # ------------------------------------------
                # Lightweight LSTM
                # ------------------------------------------

                LSTM(
                    32,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                ),

                Dropout(0.20),

                Dense(
                    16,
                    activation="relu",
                ),

                Dense(
                    self.num_classes,
                    activation="softmax",
                ),

            ]

        )

        model.compile(

            optimizer=Adam(
                learning_rate=0.001
            ),

            loss="sparse_categorical_crossentropy",

            metrics=[
                "accuracy"
            ],

        )

        self.model = model

        model.summary()

        return model

    
        # ======================================================
    # Train Model
    # ======================================================
    def train(
        self,
        epochs=200,
        batch_size=8,
    ):

        # --------------------------------------------------
        # Load Dataset
        # --------------------------------------------------

        X, y = self.load_dataset()

        # --------------------------------------------------
        # Dataset Validation
        # --------------------------------------------------

        if X.ndim != 3:
            raise ValueError(
                f"Expected X shape (N,20,64), got {X.shape}"
            )

        if y.ndim != 1:
            raise ValueError(
                f"Expected y shape (N,), got {y.shape}"
            )

        if X.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length "
                f"{self.sequence_length}, got {X.shape[1]}"
            )

        if X.shape[2] != self.feature_size:
            raise ValueError(
                f"Expected feature size "
                f"{self.feature_size}, got {X.shape[2]}"
            )

        # --------------------------------------------------
        # Train / Validation Split
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("Creating Train / Validation Split")
        print("=" * 60)

        X_train, X_val, y_train, y_val = train_test_split(

            X,

            y,

            test_size=0.10,

            random_state=42,

            stratify=y,

            shuffle=True,

        )

        print(f"Training Samples   : {len(X_train)}")
        print(f"Validation Samples : {len(X_val)}")

        # --------------------------------------------------
        # Build Model
        # --------------------------------------------------

        model = self.build_model()

        # --------------------------------------------------
        # Callbacks
        # --------------------------------------------------

        callbacks = [

            EarlyStopping(

                monitor="val_loss",

                patience=20,

                restore_best_weights=True,

                verbose=1,

            ),

            ReduceLROnPlateau(

                monitor="val_loss",

                factor=0.5,

                patience=8,

                min_lr=1e-6,

                verbose=1,

            ),

            ModelCheckpoint(

                filepath=str(MODEL_PATH),

                monitor="val_accuracy",

                save_best_only=True,

                verbose=1,

            ),

        ]

        # --------------------------------------------------
        # Training
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("Training Started")
        print("=" * 60)

        history = model.fit(

            X_train,

            y_train,

            validation_data=(
                X_val,
                y_val,
            ),

            epochs=epochs,

            batch_size=batch_size,

            callbacks=callbacks,

            shuffle=True,

            verbose=1,

        )

        print()
        print("=" * 60)
        print("Training Finished")
        print("=" * 60)

        self.model = model

        self.history = history

        # --------------------------------------------------
        # Save Training History
        # --------------------------------------------------

        history_dict = history.history

        joblib.dump(

            history_dict,

            HISTORY_PATH,

        )

        print()
        print(f"Model Saved   : {MODEL_PATH}")
        print(f"History Saved : {HISTORY_PATH}")

        return (

            model,

            history,

            X_val,

            y_val,

        )
       # ======================================================
    # Evaluate Model
    # ======================================================

    def evaluate(
        self,
        X_val,
        y_val,
    ):

        if self.model is None:

            raise RuntimeError(
                "Model has not been trained."
            )

        print()
        print("=" * 60)
        print("Evaluating Model")
        print("=" * 60)

        loss, accuracy = self.model.evaluate(
            X_val,
            y_val,
            verbose=0,
        )

        print(f"Validation Loss     : {loss:.4f}")
        print(f"Validation Accuracy : {accuracy*100:.2f}%")

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        probabilities = self.model.predict(
            X_val,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        confidences = np.max(
            probabilities,
            axis=1,
        )

        print()
        print("=" * 60)
        print("Prediction Confidence")
        print("=" * 60)

        for i in range(len(predictions)):

            print(

                f"Sample {i+1:02d} | "

                f"Actual : {LABELS[int(y_val[i])]} | "

                f"Predicted : {LABELS[int(predictions[i])]} | "

                f"Confidence : {confidences[i]*100:.2f}%"

            )

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("Confusion Matrix")
        print("=" * 60)

        cm = confusion_matrix(
            y_val,
            predictions,
        )

        print(cm)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------

        print()
        print("=" * 60)
        print("Classification Report")
        print("=" * 60)

        report = classification_report(

            y_val,

            predictions,

            target_names=[
                "J",
                "Z",
            ],

            digits=4,

        )

        print(report)

        return loss, accuracy

    # ======================================================
    # Save Training Graphs
    # ======================================================

    def save_training_plots(self):

        if self.history is None:
            return

        try:

            import matplotlib.pyplot as plt

        except ImportError:

            print(
                "matplotlib not installed. "
                "Skipping graph generation."
            )

            return

        history = self.history.history

        # --------------------------------------------------
        # Accuracy Plot
        # --------------------------------------------------

        plt.figure(figsize=(8,5))

        plt.plot(
            history["accuracy"],
            label="Training Accuracy"
        )

        plt.plot(
            history["val_accuracy"],
            label="Validation Accuracy"
        )

        plt.title("LSTM Accuracy")

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.legend()

        plt.grid(True)

        accuracy_plot = MODEL_DIR / "accuracy.png"

        plt.savefig(
            accuracy_plot,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # --------------------------------------------------
        # Loss Plot
        # --------------------------------------------------

        plt.figure(figsize=(8,5))

        plt.plot(
            history["loss"],
            label="Training Loss"
        )

        plt.plot(
            history["val_loss"],
            label="Validation Loss"
        )

        plt.title("LSTM Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend()

        plt.grid(True)

        loss_plot = MODEL_DIR / "loss.png"

        plt.savefig(
            loss_plot,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print()

        print("=" * 60)

        print("Training Graphs Saved")

        print("=" * 60)

        print(f"Accuracy Plot : {accuracy_plot}")

        print(f"Loss Plot     : {loss_plot}")
# ==========================================================
# Run Training Pipeline
# ==========================================================

def main():

    print()
    print("=" * 70)
    print("          SignLanguageAI Dynamic Gesture Trainer")
    print("                     Version 2.0")
    print("=" * 70)

    trainer = LSTMTrainer()

    try:

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        model, history, X_val, y_val = trainer.train(

            epochs=200,

            batch_size=8,

        )

        # --------------------------------------------------
        # Evaluate
        # --------------------------------------------------

        trainer.evaluate(

            X_val,

            y_val,

        )

        # --------------------------------------------------
        # Save Training Graphs
        # --------------------------------------------------

        trainer.save_training_plots()

        print()
        print("=" * 70)
        print("Training Completed Successfully")
        print("=" * 70)

        print()
        print(f"Model Saved To:")
        print(f"   {MODEL_PATH}")

        print()
        print(f"Training History:")
        print(f"   {HISTORY_PATH}")

        print()
        print("Generated Files:")

        print(f"   {MODEL_PATH.name}")
        print(f"   {HISTORY_PATH.name}")
        print("   accuracy.png")
        print("   loss.png")

        print()

        print("=" * 70)
        print("LSTM Training Summary")
        print("=" * 70)

        print("✓ Lightweight LSTM architecture")
        print("✓ Optimized for dynamic ASL J & Z")
        print("✓ Automatic EarlyStopping")
        print("✓ Adaptive learning rate scheduling")
        print("✓ Model checkpointing")
        print("✓ Confusion matrix")
        print("✓ Classification report")
        print("✓ Training graphs generated")
        print()

        print("The trained model is now ready")
        print("for Hybrid Recognition integration.")

        print("=" * 70)

    except KeyboardInterrupt:

        print()
        print("=" * 70)
        print("Training Interrupted by User")
        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("Training Failed")
        print("=" * 70)
        print(e)

        raise


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()