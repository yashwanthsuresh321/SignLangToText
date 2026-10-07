"""
Sentence Builder

Version 1.0.0
Professional State Machine
"""

import time


class SentenceBuilder:

    # ==========================================================
    # Initialization
    # ==========================================================

    def __init__(self):

        # ------------------------------------------
        # User Settings
        # ------------------------------------------

        self.hold_time = 0.55

        # Number of consecutive identical predictions
        # before starting the hold timer.

        self.required_stable_frames = 2

        # Frames required before allowing
        # another identical prediction.

        self.release_frames = 5

        # ------------------------------------------
        # Runtime
        # ------------------------------------------

        self.sentence = ""

        self.current_prediction = "-"

        self.prediction_start = None

        self.stable_count = 0

        self.waiting_for_release = False

        self.release_count = 0

        self.last_added_letter = ""

        self.letter_added = False

        self.last_added_time = 0

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self):

        self.sentence = ""

        self.current_prediction = "-"

        self.prediction_start = None

        self.stable_count = 0

        self.waiting_for_release = False

        self.release_count = 0

        self.last_added_letter = ""

        self.letter_added = False

    # ==========================================================
    # Clear Sentence
    # ==========================================================

    def clear_sentence(self):

        self.sentence = ""

    # ==========================================================
    # Backspace
    # ==========================================================

    def backspace(self):

        if len(self.sentence):

            self.sentence = self.sentence[:-1]

    # ==========================================================
    # Space
    # ==========================================================

    def add_space(self):

        if len(self.sentence) == 0:
            return

        if self.sentence.endswith(" "):
            return

        self.sentence += " "

    # ==========================================================
    # Update
    # ==========================================================

    def update(self, prediction):

        now = time.time()

        self.letter_added = False

        # ------------------------------------------------------
        # RELEASE MODE
        # ------------------------------------------------------

        if self.waiting_for_release:

            # Hand removed

            if prediction == "-":

                self.release_count += 1

            # Different gesture shown

            elif prediction != self.last_added_letter:

                self.release_count += 1

            else:

                self.release_count = 0

            # Ready for next prediction

            if self.release_count >= self.release_frames:

                self.waiting_for_release = False

                self.current_prediction = "-"

                self.prediction_start = None

                self.stable_count = 0

                self.release_count = 0

            return self.sentence

        # ------------------------------------------------------
        # No Hand
        # ------------------------------------------------------

        if prediction == "-":

            self.current_prediction = "-"

            self.prediction_start = None

            self.stable_count = 0

            return self.sentence

        # ------------------------------------------------------
        # Stable Prediction Counter
        # ------------------------------------------------------

        if prediction == self.current_prediction:

            self.stable_count += 1

        else:

            self.current_prediction = prediction

            self.stable_count = 1

            self.prediction_start = None

            return self.sentence

        # ------------------------------------------------------
        # Wait until prediction is stable
        # ------------------------------------------------------

        if self.stable_count < self.required_stable_frames:

            return self.sentence

        # ------------------------------------------------------
        # Start Hold Timer
        # ------------------------------------------------------

        if self.prediction_start is None:

            self.prediction_start = now

            return self.sentence

        # ------------------------------------------------------
        # Hold Timer
        # ------------------------------------------------------

        elapsed = now - self.prediction_start

        if elapsed < self.hold_time:

            return self.sentence
                # ------------------------------------------------------
        # Hold Complete
        # ------------------------------------------------------

        if prediction == "SPACE":
            self.add_space()

        elif prediction == "BACKSPACE":
            self.backspace()

        elif prediction == "CLEAR":
            self.clear_sentence()

        else:
            self.sentence += prediction

        self.last_added_letter = prediction

        self.last_added_time = now

        self.letter_added = True

        # Enter release mode
        self.waiting_for_release = True

        self.release_count = 0

        # Reset tracking for next letter
        self.current_prediction = prediction

        self.prediction_start = None

        self.stable_count = 0

        return self.sentence

    # ==========================================================
    # Sentence
    # ==========================================================

    def get_sentence(self):

        return self.sentence

    # ==========================================================
    # Hold Progress
    # ==========================================================

    def get_progress(self):

        # Waiting for prediction to stabilize
        if self.stable_count < self.required_stable_frames:

            if self.required_stable_frames == 0:
                return 0.0

            return self.stable_count / self.required_stable_frames

        # Hold timer hasn't started yet
        if self.prediction_start is None:

            return 0.0

        progress = (
            time.time() - self.prediction_start
        ) / self.hold_time

        return min(max(progress, 0.0), 1.0)

    # ==========================================================
    # Letter Added?
    # ==========================================================

    def is_letter_added(self):

        return self.letter_added

    # ==========================================================
    # Last Added Letter
    # ==========================================================

    def get_last_added_letter(self):

        return self.last_added_letter