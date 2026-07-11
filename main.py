"""
SignLanguageAI
Version 0.6.0

Main Application
"""

import time
import cv2
import mediapipe as mp

from src.config import Config

from src.camera.camera import Camera

from src.utils.keyboard import Keyboard
from src.utils.ui import UI
from src.gestures.gesture_detector import GestureDetector
from src.dataset.dataset_collector import DatasetCollector

from src.detection.hand_detector import HandDetector
from src.detection.hand_drawer import HandDrawer

from src.prediction.predictor import Predictor
from src.sentence.sentence_builder import SentenceBuilder

# ==========================================================
# Initialization
# ==========================================================

camera = Camera()
camera.create_window()

keyboard = Keyboard()

ui = UI()

collector = DatasetCollector()

detector = HandDetector(
    str(Config.HAND_LANDMARKER_MODEL)
)

drawer = HandDrawer()

# ----------------------------------------------------------
# AI Predictor
# ----------------------------------------------------------

predictor = Predictor(
    model_path=Config.SIGN_MODEL_PATH,
    scaler_path=Config.SCALER_PATH,
    label_encoder_path=Config.LABEL_ENCODER_PATH
)

resolution = camera.get_resolution()
sentence_builder = SentenceBuilder()
gesture_detector = GestureDetector()
# ----------------------------------------------------------
# MediaPipe VIDEO Timer
# ----------------------------------------------------------

start_time = time.perf_counter()
sentence = ""

# ==========================================================
# Main Loop
# ==========================================================

while camera.is_opened():

    # ----------------------------------------------
    # Camera
    # ----------------------------------------------

    frame = camera.read()

    if frame is None:
        break

    # ----------------------------------------------
    # Update Collector State
    # ----------------------------------------------

    collector.update()

    # ----------------------------------------------
    # MediaPipe Image
    # ----------------------------------------------

    # ----------------------------------------------
    # Improve frame for hand detection
    # ----------------------------------------------

    enhanced_frame = cv2.convertScaleAbs(
        frame,
        alpha=1.15,   # Contrast
        beta=15       # Brightness
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

    result = detector.detect(
        mp_image,
        timestamp_ms
    )

    # ----------------------------------------------
    # Detection Variables
    # ----------------------------------------------

    hand_count = 0

    handedness = "-"

    confidence = 0.0

    detected_landmarks = None

    detected_hand = "-"

    # --------------------------
    # AI Prediction Variables
    # --------------------------

    prediction = "-"

    prediction_confidence = 0.0
    gesture_command = None
    # ----------------------------------------------
    # Hand Detection
    # ----------------------------------------------

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

            # Mirror correction

            if Config.MIRROR_CAMERA:

                if handedness == "Left":

                    handedness = "Right"

                elif handedness == "Right":

                    handedness = "Left"

            detected_hand = handedness
        # ------------------------------------------
        # Command Gesture Detection
        # ------------------------------------------

        gesture_command = gesture_detector.detect(

            detected_landmarks,

            detected_hand
        )
        # ------------------------------------------
        # Live AI Prediction
        # ------------------------------------------

        if Config.ENABLE_PREDICTION:

            prediction, prediction_confidence = (

                predictor.predict_live(

                    detected_landmarks,

                    detected_hand,

                    Config.CONFIDENCE_THRESHOLD

                )

            )

            prediction_confidence *= 100
            sentence = sentence_builder.update(
                prediction
            )
            if gesture_command is not None:

                print(

                    f"Command Detected : {gesture_command}"
                )
            hold_progress = sentence_builder.get_progress()

            letter_added = sentence_builder.is_letter_added()

            last_added_letter = (
                sentence_builder.get_last_added_letter()
            )
    else:

        sentence = sentence_builder.update("-")
        hold_progress = 0.0

        letter_added = False

        last_added_letter = ""
    # ----------------------------------------------
    # Keyboard Events
    # ----------------------------------------------

    event = keyboard.get_event()

    if event is not None:

        if event["type"] == "QUIT":

            break

        elif event["type"] == "LETTER":

            if detected_landmarks is not None:

                collector.save_sample(

                    label=event["key"],

                    hand=detected_hand,

                    landmarks=detected_landmarks

                )

    # ----------------------------------------------
    # Dataset Progress
    # ----------------------------------------------

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

    # ======================================================
    # Draw User Interface
    # ======================================================

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

        session_count=session_count

    )

    # ======================================================
    # Display Window
    # ======================================================

    cv2.imshow(

        Config.WINDOW_NAME,

        frame

    )


# ==========================================================
# Cleanup
# ==========================================================

camera.release()

cv2.destroyAllWindows()


# ==========================================================
# Dataset Summary
# ==========================================================

print()

print("=" * 60)
print("          SIGNLANGUAGEAI DATASET SUMMARY")
print("=" * 60)

if len(collector.sample_count) == 0:

    print("No samples collected.")

else:

    total = collector.get_total_samples()

    print()

    print(f"{'Letter':<10}{'Dataset':>12}{'Session':>12}")

    print("-" * 60)

    for label in sorted(collector.sample_count.keys()):

        dataset_count = collector.get_dataset_count(label)

        session_count = collector.get_session_count(label)

        print(

            f"{label:<10}"

            f"{dataset_count:>12}"

            f"{session_count:>12}"

        )

    print("-" * 60)

    print(

        f"{'TOTAL':<10}"

        f"{total:>12}"

    )

print("=" * 60)

print("\nThank you for using SignLanguageAI.\n")