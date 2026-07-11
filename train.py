"""
SignLanguageAI
Version 0.5.0

Neural Network Training Script
"""

from pathlib import Path

from src.config import Config

from src.training.dataset import DatasetLoader
from src.training.preprocessing import DataPreprocessor
from src.training.model import SignLanguageModel
from src.training.trainer import ModelTrainer
from src.training.evaluator import ModelEvaluator


# ==========================================================
# Paths
# ==========================================================

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "sign_language_model.keras"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"


# ==========================================================
# Main
# ==========================================================

def main():

    print("\n" + "=" * 60)
    print("        SIGNLANGUAGEAI MODEL TRAINING")
    print("=" * 60)

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    loader = DatasetLoader(Config.DATASET_PATH)

    X, y = loader.prepare()

    # ------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------

    preprocessor = DataPreprocessor()

    X_train, X_test, y_train, y_test = preprocessor.prepare(
        X,
        y
    )

    # ------------------------------------------------------
    # Build Model
    # ------------------------------------------------------

    num_classes = len(preprocessor.label_encoder.classes_)

    builder = SignLanguageModel(
        input_shape=X_train.shape[1],
        num_classes=num_classes
    )

    model = builder.build()

    builder.compile()

    builder.summary()

    # ------------------------------------------------------
    # Train Model
    # ------------------------------------------------------

    trainer = ModelTrainer(
        model=model,
        model_path=MODEL_PATH
    )

    history = trainer.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=100,
        batch_size=16
    )

    trainer.summary()

    # ------------------------------------------------------
    # Save Model
    # ------------------------------------------------------

    trainer.save()

    preprocessor.save_label_encoder(
        LABEL_ENCODER_PATH
    )

    preprocessor.save_scaler(
        SCALER_PATH
    )

    # ------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------

    evaluator = ModelEvaluator(
        model,
        preprocessor.label_encoder
    )

    evaluator.evaluate(
        X_test,
        y_test
    )

    evaluator.plot_history(
        history
    )

    print("\n" + "=" * 60)
    print("Training Completed Successfully")
    print("=" * 60)

    print(f"\nModel Saved          : {MODEL_PATH}")
    print(f"Label Encoder Saved  : {LABEL_ENCODER_PATH}")
    print(f"Feature Scaler Saved : {SCALER_PATH}")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()