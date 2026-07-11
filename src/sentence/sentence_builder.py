"""
Sentence Builder
----------------
Converts stable letter predictions into a sentence.

Version 0.6.2 (State Machine)
"""

import time


class SentenceBuilder:

    def __init__(self):

        # ==========================================
        # Settings
        # ==========================================

        self.hold_time = 0.8

        # ==========================================
        # Runtime
        # ==========================================

        self.current_prediction = "-"

        self.prediction_start = None

        self.waiting_for_release = False

        self.sentence = ""
        # ==========================================
        # UI Feedback
        # ==========================================

        self.last_added_letter = ""

        self.letter_added = False

        self.last_added_time = 0

    # ==========================================================
    # Reset Everything
    # ==========================================================

    def reset(self):

        self.current_prediction = "-"

        self.prediction_start = None

        self.waiting_for_release = False

        self.sentence = ""

    # ==========================================================
    # Clear Sentence
    # ==========================================================

    def clear_sentence(self):

        self.sentence = ""

    # ==========================================================
    # Backspace
    # ==========================================================

    def backspace(self):

        if len(self.sentence) > 0:

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

        # ------------------------------------------
        # WAIT FOR RELEASE STATE
        # ------------------------------------------

        if self.waiting_for_release:

            if prediction == "-":

                self.waiting_for_release = False

                self.current_prediction = "-"

                self.prediction_start = None

            return self.sentence

        # ------------------------------------------
        # No prediction
        # ------------------------------------------

        if prediction == "-":

            self.current_prediction = "-"

            self.prediction_start = None

            return self.sentence

        # ------------------------------------------
        # New prediction
        # ------------------------------------------

        if prediction != self.current_prediction:

            self.current_prediction = prediction

            self.prediction_start = now

            return self.sentence

        # ------------------------------------------
        # Hold timer
        # ------------------------------------------

        elapsed = now - self.prediction_start

        if elapsed >= self.hold_time:

            self.sentence += prediction
            self.last_added_letter = prediction

            self.letter_added = True

            self.last_added_time = now

            self.waiting_for_release = True

            self.current_prediction = "-"

            self.prediction_start = None

        return self.sentence

    # ==========================================================
    # Sentence
    # ==========================================================

    def get_sentence(self):

        return self.sentence

    # ==========================================================
    # Hold Progress
    # ==========================================================
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
    def get_progress(self):

        if self.prediction_start is None:

            return 0.0

        progress = (
            (time.time() - self.prediction_start)
            / self.hold_time
        )

        return min(progress, 1.0)