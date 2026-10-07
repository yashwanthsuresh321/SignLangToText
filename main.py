"""
SignLanguageAI
Version 0.8.1

Main Application

Silent terminal version:
- Keeps application behavior unchanged
- Suppresses TensorFlow / MediaPipe native logs
- Suppresses startup model messages
- Keeps real Python exceptions visible
"""

import io
import logging
import os
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout

# These must be set before TensorFlow / MediaPipe are imported.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"


# TensorFlow also emits some messages through Python's logging system.
# Suppress only TensorFlow WARNING/INFO messages; real ERROR messages
# remain visible.
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


@contextmanager
def _silent_native_stderr():
    """
    Temporarily redirect only the OS-level stderr handle.

    Unlike the previous implementation, stdout is never redirected,
    so Python's normal print() remains valid on Windows.
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass

    stderr_fd = sys.stderr.fileno()
    saved_stderr = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), stderr_fd)
            yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass

        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)


# Third-party imports can emit native TensorFlow / MediaPipe messages.
# Suppress only OS-level stderr while they initialize.
with _silent_native_stderr():

    import cv2
    import mediapipe as mp
    import tensorflow as tf

    # TensorFlow may configure its logger during import.
    tf.get_logger().setLevel(logging.ERROR)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    from src.config import Config

    from src.camera.camera import Camera

    from src.utils.keyboard import Keyboard
    from src.ui.ui import UI

    from src.dataset.dataset_collector import DatasetCollector

    from src.detection.hand_detector import HandDetector
    from src.detection.hand_drawer import HandDrawer

    from src.prediction.hybrid_predictor import HybridPredictor

    from src.dynamic.sequence_collector import SequenceCollector
    from src.dynamic.sequence_dataset import SequenceDataset

    from src.sentence.sentence_builder import SentenceBuilder


# ==========================================================
# Dynamic Dataset Configuration
# ==========================================================

DYNAMIC_DATASET_PATH = os.path.join(
    "data",
    "datasets",
    "dynamic_dataset.npz"
)


# ==========================================================
# Initialization
# ==========================================================

camera = Camera()
camera.create_window()

keyboard = Keyboard()

ui = UI()

collector = DatasetCollector()

# Detector initialization can emit native MediaPipe messages.
with _silent_native_stderr():

    detector = HandDetector(
        str(Config.HAND_LANDMARKER_MODEL)
    )

    drawer = HandDrawer()


# Predictor startup uses normal Python print() statements.
# Redirect only Python stdout here; no OS handles are modified.
with redirect_stdout(io.StringIO()):

    predictor = HybridPredictor(

        model_path=Config.SIGN_MODEL_PATH,

        scaler_path=Config.SCALER_PATH,

        label_encoder_path=Config.LABEL_ENCODER_PATH,

        dynamic_model_path="data/models/dynamic_sign_model.keras",

        dynamic_threshold=0.70,

        static_threshold=Config.CONFIDENCE_THRESHOLD,

    )

resolution = camera.get_resolution()

sentence_builder = SentenceBuilder()

dynamic_collector = SequenceCollector()
dynamic_dataset = SequenceDataset()

# ==========================================================
# Automatically Load Existing Dynamic Dataset
# ==========================================================

# Load the existing dynamic dataset silently.
# Dataset behavior is unchanged; only terminal output is removed.
dynamic_dataset.load(DYNAMIC_DATASET_PATH)


# ==========================================================
# Dynamic UI State
# ==========================================================

dynamic_label = "-"

dynamic_frames = 0

dynamic_target = 20

dynamic_recording = False

dynamic_save_message = ""

dynamic_last_saved_label = ""


# ==========================================================
# MediaPipe Timer
# ==========================================================

start_time = time.perf_counter()
sentence = ""

hold_progress = 0.0
letter_added = False
last_added_letter = ""

prediction = "-"
prediction_confidence = 0.0
prediction_source = "STATIC"




# ==========================================================
# Main Loop
# ==========================================================

while camera.is_opened():

    # ------------------------------------------------------
    # Camera
    # ------------------------------------------------------

    frame = camera.read()

    if frame is None:
        break

    # ------------------------------------------------------
    # Update Dataset Collector
    # ------------------------------------------------------

    collector.update()

    # ------------------------------------------------------
    # Improve Frame Quality
    # ------------------------------------------------------

    enhanced_frame = cv2.convertScaleAbs(
        frame,
        alpha=1.15,
        beta=15
    )

    frame_rgb = cv2.cvtColor(
        enhanced_frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    # MediaPipe may emit native C++ warnings to stderr.
    # Suppress those messages without touching Python stdout.
    with _silent_native_stderr():

        result = detector.detect(
            mp_image,
            timestamp_ms
        )

    # ------------------------------------------------------
    # Detection Variables
    # ------------------------------------------------------

    hand_count = 0

    handedness = "-"

    confidence = 0.0

    detected_landmarks = None

    detected_hand = "-"

    prediction = "-"

    prediction_confidence = 0.0
    prediction_source = "STATIC"
    # ------------------------------------------------------
    # Hand Detection
    # ------------------------------------------------------

    if result.hand_landmarks:

        hand_count = len(result.hand_landmarks)

        detected_landmarks = result.hand_landmarks[0]

        for hand_landmarks in result.hand_landmarks:

            frame = drawer.draw(
                frame,
                hand_landmarks
            )

        if result.handedness:

            category = result.handedness[0][0]

            handedness = category.category_name

            confidence = category.score * 100

            if Config.MIRROR_CAMERA:

                if handedness == "Left":
                    handedness = "Right"

                elif handedness == "Right":
                    handedness = "Left"

            detected_hand = handedness

        # --------------------------------------------------
        # Static Prediction
        #
        # DO NOT MODIFY THIS SECTION
        # --------------------------------------------------
        if Config.ENABLE_PREDICTION and not dynamic_recording:
            try:
                hybrid_result = predictor.predict(
                    detected_landmarks,
                    detected_hand,
                )
                prediction = hybrid_result["prediction"]

                prediction_confidence = (
                    hybrid_result["confidence"] * 100
                )

                prediction_source = hybrid_result["source"]

                sentence = sentence_builder.update(
                    prediction
                )

                hold_progress = (
                    sentence_builder.get_progress()
                )

                letter_added = (
                    sentence_builder.is_letter_added()
                )

                last_added_letter = (
                    sentence_builder.get_last_added_letter()
                )

            except Exception as e:
                print("\nHYBRID PREDICTOR ERROR")
                print(type(e).__name__)
                print(e)

                raise

        else:
            sentence = sentence_builder.update("-")

            hold_progress = 0.0

            letter_added = False

            last_added_letter = ""
        # ==========================================================
    # Keyboard Events
    # ==========================================================

    event = keyboard.get_event()

    if event is not None:

        # ------------------------------------------------------
        # Quit
        # ------------------------------------------------------

        if event["type"] == "QUIT":

            break

        # ------------------------------------------------------
        # Static Dataset Collection
        #
        # UNCHANGED
        # ------------------------------------------------------

        elif event["type"] in ["LETTER", "COMMAND"]:

            if detected_landmarks is not None:

                collector.save_sample(

                    label=event["key"],

                    hand=detected_hand,

                    landmarks=detected_landmarks

                )

        # ------------------------------------------------------
        # Start Dynamic Recording
        # ------------------------------------------------------

        elif event["type"] == "DYNAMIC_RECORD":

            dynamic_label = event["label"]

            dynamic_recording = True

            dynamic_frames = 0

            dynamic_save_message = ""

            try:

                dynamic_collector.start(dynamic_label)

            except Exception as e:

                print()
                print("[Dynamic] Failed to start recording.")
                print(e)

                dynamic_recording = False

        # ------------------------------------------------------
        # Manual Save
        # ------------------------------------------------------

        elif event["type"] == "SAVE_DYNAMIC_DATASET":

            try:

                dynamic_dataset.save(
                    DYNAMIC_DATASET_PATH
                )

            except Exception as e:

                print()
                print("[Dynamic] Save failed.")
                print(e)

    # ==========================================================
    # Dynamic Sequence Recording
    # ==========================================================

    if dynamic_recording and detected_landmarks is not None:

        try:

            # ----------------------------------------------
            # Extract the same 64 features used for training
            # ----------------------------------------------

            features = predictor.extract_features(

                detected_landmarks,

                detected_hand

            )

            # ----------------------------------------------
            # Add frame
            # ----------------------------------------------

            dynamic_collector.update(features)

            dynamic_frames = (
                dynamic_collector.frames_collected()
            )

            # ----------------------------------------------
            # Completed sequence?
            # ----------------------------------------------

            if dynamic_collector.is_complete():

                sequence, label = (
                    dynamic_collector.get_sequence()
                )

                dynamic_dataset.add_sequence(

                    sequence,

                    label

                )

                # ------------------------------------------
                # Auto Save
                # ------------------------------------------

                dynamic_dataset.save(
                    DYNAMIC_DATASET_PATH
                )

                counts = dynamic_dataset.label_counts()

                dynamic_last_saved_label = label

                dynamic_save_message = (
                    f"{label} saved successfully"
                )

                dynamic_recording = False

                dynamic_frames = 0

        except Exception as e:

            print()

            print("[Dynamic] Recording Error")

            print(e)

            dynamic_recording = False

            dynamic_frames = 0

    # ==========================================================
    # Static Dataset Progress
    # ==========================================================

    if collector.current_label != "-":

        (

            current_count,

            target_count,

            progress

        ) = collector.get_progress(

            collector.current_label

        )

        session_count = (

            collector.get_session_count(

                collector.current_label

            )

        )

    else:

        current_count = 0

        target_count = Config.TARGET_SAMPLES

        progress = 0

        session_count = 0

    # ==========================================================
    # Dynamic Dataset Statistics
    # ==========================================================

    counts = dynamic_dataset.label_counts()

    dynamic_j_count = counts["J"]

    dynamic_z_count = counts["Z"]

    dynamic_goal = 150

    if dynamic_label == "J":

        dynamic_collected = dynamic_j_count

    elif dynamic_label == "Z":

        dynamic_collected = dynamic_z_count

    else:

        dynamic_collected = 0
        # ==========================================================
    # Draw User Interface
    # ==========================================================

    frame = ui.draw_overlay(

        frame=frame,

        hand_count=hand_count,

        handedness=handedness,

        confidence=confidence,

        resolution=resolution,

        prediction=prediction,

        prediction_confidence=prediction_confidence,

        sentence=sentence,

        hold_progress=hold_progress,

        letter_added=letter_added,

        last_added_letter=last_added_letter,

        current_label=collector.current_label,

        current_count=current_count,

        target_count=target_count,

        progress=progress,

        status=collector.last_status,

        session_count=session_count,

        # --------------------------------------------------
        # Dynamic Recognition UI
        # --------------------------------------------------

        dynamic_label=dynamic_label,

        dynamic_frames=dynamic_frames,

        dynamic_target=dynamic_target,

        dynamic_recording=dynamic_recording,

        dynamic_collected=dynamic_collected,

        dynamic_goal=dynamic_goal,

        dynamic_j_count=dynamic_j_count,

        dynamic_z_count=dynamic_z_count,

    )

    # ==========================================================
    # Display Window
    # ==========================================================

    cv2.imshow(

        Config.WINDOW_NAME,

        frame

    )

# ==============================================================
# Cleanup
# ==============================================================

camera.release()

cv2.destroyAllWindows()

# ==============================================================
# Save Dynamic Dataset Before Exit
# ==============================================================

try:

    dynamic_dataset.save(
        DYNAMIC_DATASET_PATH
    )

except Exception:
    # Preserve silent operation on normal shutdown.
    pass

