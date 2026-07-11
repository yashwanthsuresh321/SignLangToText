"""
SignLanguageAI
Version 0.6.0

Project Configuration
"""

from pathlib import Path


class Config:

    # ==========================
    # Project
    # ==========================

    WINDOW_NAME = "SignLanguageAI"

    # ==========================
    # Camera
    # ==========================

    CAMERA_INDEX = 0

    FRAME_WIDTH = 1280

    FRAME_HEIGHT = 720

    MIRROR_CAMERA = True

    # ==========================
    # Dataset
    # ==========================

    DATASET_PATH = Path(
        "data/csv/sign_language_dataset.csv"
    )

    TARGET_SAMPLES = 150

    SAVE_COOLDOWN = 1.0

    MOVEMENT_THRESHOLD = 0.02
        # ==========================
    # Labels
    # ==========================

    LETTER_LABELS = [

        "A","B","C","D","E","F","G","H","I","J",

        "K","L","M","N","O","P","Q","R","S","T",

        "U","V","W","X","Y","Z"

    ]

    COMMAND_LABELS = [

        "SPACE",

        "BACKSPACE",

        "CLEAR"

    ]

    ALL_LABELS = LETTER_LABELS + COMMAND_LABELS

    # ==========================
    # MediaPipe
    # ==========================

    HAND_LANDMARKER_MODEL = Path(
        "data/models/hand_landmarker.task"
    )

    MAX_HANDS = 2

    DETECTION_CONFIDENCE = 0.5

    TRACKING_CONFIDENCE = 0.5

    PRESENCE_CONFIDENCE = 0.5

    # ==========================
    # AI Model
    # ==========================

    SIGN_MODEL_PATH = Path(
        "data/models/sign_language_model.keras"
    )

    SCALER_PATH = Path(
        "data/models/feature_scaler.pkl"
    )

    LABEL_ENCODER_PATH = Path(
        "data/models/label_encoder.pkl"
    )

    CONFIDENCE_THRESHOLD = 0.75

    # ==========================
    # Prediction
    # ==========================

    ENABLE_PREDICTION = True

    SMOOTHING_WINDOW = 5

    MIN_CONSISTENT_PREDICTIONS = 3

    # ==========================
    # UI
    # ==========================

    FONT_SCALE = 0.65

    FONT_THICKNESS = 2