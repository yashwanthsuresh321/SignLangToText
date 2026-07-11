"""
SignLanguageAI
Version 0.5.0

Model Evaluator
"""

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:

    def __init__(
        self,
        model,
        label_encoder
    ):

        self.model = model

        self.label_encoder = label_encoder

    # ==========================================================
    # Evaluate
    # ==========================================================

    def evaluate(

        self,

        X_test,

        y_test

    ):

        print()

        print("=" * 60)

        print("           MODEL EVALUATION")

        print("=" * 60)

        probabilities = self.model.predict(

            X_test,

            verbose=0

        )

        predictions = probabilities.argmax(

            axis=1

        )

        accuracy = accuracy_score(

            y_test,

            predictions

        )

        print(

            f"\nAccuracy : {accuracy * 100:.2f}%"

        )

        print()

        print("=" * 60)

        print("CLASSIFICATION REPORT")

        print("=" * 60)

        print(

            classification_report(

                y_test,

                predictions,

                target_names=self.label_encoder.classes_

            )

        )

        print("=" * 60)

        print()

        print("CONFUSION MATRIX")

        print("=" * 60)

        cm = confusion_matrix(

            y_test,

            predictions

        )

        print(cm)

        print("=" * 60)

        return predictions

    # ==========================================================
    # Plot Training History
    # ==========================================================

    def plot_history(

        self,

        history

    ):

        if history is None:

            return

        plt.figure(

            figsize=(8,5)

        )

        plt.plot(

            history.history["accuracy"],

            label="Training Accuracy"

        )

        plt.plot(

            history.history["val_accuracy"],

            label="Validation Accuracy"

        )

        plt.title(

            "Model Accuracy"

        )

        plt.xlabel(

            "Epoch"

        )

        plt.ylabel(

            "Accuracy"

        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.show()

        plt.figure(

            figsize=(8,5)

        )

        plt.plot(

            history.history["loss"],

            label="Training Loss"

        )

        plt.plot(

            history.history["val_loss"],

            label="Validation Loss"

        )

        plt.title(

            "Model Loss"

        )

        plt.xlabel(

            "Epoch"

        )

        plt.ylabel(

            "Loss"

        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.show()

    # ==========================================================
    # Predict One Sample
    # ==========================================================

    def predict(

        self,

        sample

    ):

        probabilities = self.model.predict(

            sample,

            verbose=0

        )

        prediction = probabilities.argmax(

            axis=1

        )

        letter = self.label_encoder.inverse_transform(

            prediction

        )[0]

        confidence = probabilities.max()

        return (

            letter,

            confidence

        )