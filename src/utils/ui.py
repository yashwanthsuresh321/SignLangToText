"""
SignLanguageAI
Version 0.6.0

Professional User Interface
"""

import cv2
import time


class UI:

    def __init__(self):

        self.previous_time = time.time()

        # ==================================================
        # Theme Colors (BGR)
        # ==================================================

        self.BACKGROUND = (35, 35, 35)

        self.HEADER = (45, 45, 45)

        self.WHITE = (255, 255, 255)

        self.BLACK = (0, 0, 0)

        self.GREEN = (0, 220, 0)

        self.YELLOW = (0, 255, 255)

        self.RED = (0, 0, 255)

        self.ORANGE = (0, 165, 255)

        self.CYAN = (255, 255, 0)

        self.GRAY = (90, 90, 90)

        self.LIGHT_GRAY = (180, 180, 180)

        self.PROGRESS_BG = (70, 70, 70)

        self.PROGRESS_FILL = (0, 200, 0)

    # ======================================================
    # FPS
    # ======================================================

    def calculate_fps(self):

        current_time = time.time()

        fps = 1 / max(

            current_time - self.previous_time,

            0.0001

        )

        self.previous_time = current_time

        return int(fps)

    # ======================================================
    # Status Color
    # ======================================================

    def get_status_color(
        self,
        status
    ):

        status = status.lower()

        if "saved" in status:

            return self.GREEN

        if "ready" in status:

            return self.GREEN

        if "cooldown" in status:

            return self.YELLOW

        if "move" in status:

            return self.ORANGE

        if "complete" in status:

            return self.CYAN

        return self.WHITE

    # ======================================================
    # Draw Header
    # ======================================================

    def draw_header(

        self,

        frame,

        fps,

        hand_count,

        handedness,

        confidence,

        resolution

    ):

        h, w = frame.shape[:2]

        cv2.rectangle(

            frame,

            (0, 0),

            (w, 48),

            self.HEADER,

            -1

        )

        left_text = (

            f"FPS : {fps}"

            f"    Hands : {hand_count}"

        )

        right_text = (

            f"Hand : {handedness}"

            f"    Detection : {confidence:.1f}%"

            f"    Resolution : {resolution}"

        )

        cv2.putText(

            frame,

            left_text,

            (15, 31),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            self.WHITE,

            2

        )

        text_size = cv2.getTextSize(

            right_text,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            2

        )[0]

        cv2.putText(

            frame,

            right_text,

            (

                w - text_size[0] - 20,

                31

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            self.WHITE,

            2

        )

        return frame

    # ======================================================
    # Bottom Panel
    # ======================================================

    def draw_bottom_panel(

        self,

        frame

    ):

        h, w = frame.shape[:2]

        panel_height = 180

        panel_top = h - panel_height

        cv2.rectangle(

            frame,

            (0, panel_top),

            (w, h),

            self.BACKGROUND,

            -1

        )

        cv2.line(

            frame,

            (0, panel_top),

            (w, panel_top),

            self.GRAY,

            2

        )

        return frame
        # ======================================================
    # Dataset Information
    # ======================================================

    def draw_dataset_info(

        self,

        frame,

        current_label,

        current_count,

        target_count,

        session_count

    ):

        h, w = frame.shape[:2]

        panel_top = h - 180

        remaining = max(

            target_count - current_count,

            0

        )

        cv2.putText(

            frame,

            "DATASET",

            (20, panel_top + 28),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.72,

            self.CYAN,

            2

        )

        cv2.putText(

            frame,

            f"Letter : {current_label}",

            (20, panel_top + 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.62,

            self.WHITE,

            2

        )

        cv2.putText(

            frame,

            f"Collected : {current_count}/{target_count}",

            (20, panel_top + 92),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.62,

            self.WHITE,

            2

        )

        cv2.putText(

            frame,

            f"Remaining : {remaining}",

            (20, panel_top + 124),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.62,

            self.WHITE,

            2

        )

        cv2.putText(

            frame,

            f"Session : {session_count}",

            (20, panel_top + 156),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.62,

            self.WHITE,

            2

        )

        return frame

    # ======================================================
    # Progress Bar
    # ======================================================

    def draw_progress_bar(

        self,

        frame,

        progress,

        current_count,

        target_count

    ):

        h, w = frame.shape[:2]

        panel_top = h - 180

        bar_x = 340

        bar_y = panel_top + 60

        bar_width = w - 380

        bar_height = 34

        cv2.rectangle(

            frame,

            (bar_x, bar_y),

            (bar_x + bar_width, bar_y + bar_height),

            self.PROGRESS_BG,

            -1

        )

        filled = int(

            (progress / 100.0)

            * bar_width

        )

        cv2.rectangle(

            frame,

            (bar_x, bar_y),

            (bar_x + filled, bar_y + bar_height),

            self.PROGRESS_FILL,

            -1

        )

        cv2.rectangle(

            frame,

            (bar_x, bar_y),

            (bar_x + bar_width, bar_y + bar_height),

            self.WHITE,

            2

        )

        progress_text = (

            f"{current_count}/{target_count}"

            f" ({progress:.1f}%)"

        )

        text_size = cv2.getTextSize(

            progress_text,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            2

        )[0]

        cv2.putText(

            frame,

            progress_text,

            (

                bar_x +

                (bar_width - text_size[0]) // 2,

                bar_y + 24

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            self.BLACK,

            2

        )

        return frame

    # ======================================================
    # Status Panel
    # ======================================================

    def draw_status_panel(

        self,

        frame,

        status

    ):

        h, w = frame.shape[:2]

        panel_top = h - 180

        color = self.get_status_color(

            status

        )

        cv2.putText(

            frame,

            "STATUS",

            (340, panel_top + 120),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.72,

            self.CYAN,

            2

        )

        cv2.putText(

            frame,

            status,

            (340, panel_top + 155),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.70,

            color,

            2

        )

        return frame
        # ======================================================
    # Prediction Panel
    # ======================================================

    def draw_prediction_panel(

        self,

        frame,

        prediction,

        prediction_confidence

    ):

        h, w = frame.shape[:2]

        panel_top = h - 180

        x = w - 320

        cv2.putText(

            frame,

            "PREDICTION",

            (x, panel_top + 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.72,

            self.CYAN,

            2

        )

        # ----------------------------------------------
        # Prediction Color
        # ----------------------------------------------

        if prediction == "-":

            color = self.LIGHT_GRAY

        elif prediction_confidence >= 90:

            color = self.GREEN

        elif prediction_confidence >= 75:

            color = self.YELLOW

        else:

            color = self.RED

        # ----------------------------------------------
        # Predicted Letter
        # ----------------------------------------------

        cv2.putText(

            frame,

            str(prediction),

            (x, panel_top + 78),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.95,

            color,

            2

        )

        # ----------------------------------------------
        # Confidence
        # ----------------------------------------------

        cv2.putText(

            frame,

            f"Confidence : {prediction_confidence:.1f}%",

            (x, panel_top + 108),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            self.WHITE,

            2

        )

        return frame
        # ======================================================
    # Hold Progress
    # ======================================================

    def draw_hold_progress(

        self,

        frame,

        progress

    ):

        h, w = frame.shape[:2]

        panel_top = h - 180

        x = w - 320

        y = panel_top + 125

        width = 200

        height = 18

        # Background

        cv2.rectangle(

            frame,

            (x, y),

            (x + width, y + height),

            self.PROGRESS_BG,

            -1

        )

        filled = int(

            width * progress

        )

        cv2.rectangle(

            frame,

            (x, y),

            (x + filled, y + height),

            self.GREEN,

            -1

        )

        cv2.rectangle(

            frame,

            (x, y),

            (x + width, y + height),

            self.WHITE,

            2

        )

        cv2.putText(

            frame,

            "Hold",

            (x, y - 8),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            self.WHITE,

            1

        )

        return frame
        # ======================================================
    # Letter Added
    # ======================================================

    def draw_letter_added(

        self,

        frame,

        letter_added,

        last_added_letter

    ):

        if not letter_added:

            return frame

        h, w = frame.shape[:2]

        panel_top = h - 180

        x = w - 320

        cv2.putText(

            frame,

            f"✓ Added : {last_added_letter}",

            (x, panel_top + 170),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.62,

            self.GREEN,

            2

        )

        return frame
    # ======================================================
    # Sentence Panel
    # ======================================================

    def draw_sentence_panel(

        self,

        frame,

        sentence

    ):

        h, w = frame.shape[:2]

        panel_top = h - 180

        x = w - 320

        cv2.putText(

            frame,

            "SENTENCE",

            (x, panel_top + 140),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.72,

            self.CYAN,

            2

        )

        if sentence == "":

            sentence = "-"

        cv2.putText(

            frame,

            sentence,

            (x, panel_top + 172),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.68,

            self.WHITE,

            2

        )

        return frame

    # ======================================================
    # Decorative Separators
    # ======================================================

    def draw_separators(

        self,

        frame

    ):

        h, w = frame.shape[:2]

        panel_top = h - 180

        # Left separator

        cv2.line(

            frame,

            (300, panel_top + 15),

            (300, h - 15),

            self.GRAY,

            2

        )

        # Right separator

        cv2.line(

            frame,

            (w - 340, panel_top + 15),

            (w - 340, h - 15),

            self.GRAY,

            2

        )

        return frame
        # ======================================================
    # Main Draw Function
    # ======================================================

    def draw_overlay(

        self,

        frame,

        hand_count=0,

        handedness="-",

        confidence=0.0,

        resolution="",

        prediction="-",

        prediction_confidence=0.0,

        sentence="",
        
        hold_progress=0.0,

        letter_added=False,

        last_added_letter="",

        current_label="-",

        current_count=0,

        target_count=300,

        progress=0,

        status="Ready",

        session_count=0

    ):

        # ------------------------------------------
        # FPS
        # ------------------------------------------

        fps = self.calculate_fps()

        # ------------------------------------------
        # Header
        # ------------------------------------------

        frame = self.draw_header(

            frame,

            fps,

            hand_count,

            handedness,

            confidence,

            resolution

        )

        # ------------------------------------------
        # Bottom Panel
        # ------------------------------------------

        frame = self.draw_bottom_panel(

            frame

        )

        # ------------------------------------------
        # Dataset Information
        # ------------------------------------------

        frame = self.draw_dataset_info(

            frame,

            current_label,

            current_count,

            target_count,

            session_count

        )

        # ------------------------------------------
        # Progress Bar
        # ------------------------------------------

        frame = self.draw_progress_bar(

            frame,

            progress,

            current_count,

            target_count

        )

        # ------------------------------------------
        # Status
        # ------------------------------------------

        frame = self.draw_status_panel(

            frame,

            status

        )

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        frame = self.draw_prediction_panel(

            frame,

            prediction,

            prediction_confidence

        )
        frame = self.draw_hold_progress(

            frame,

            hold_progress
 )

        # ------------------------------------------
        # Sentence
        # ------------------------------------------

        frame = self.draw_sentence_panel(

            frame,

            sentence

        )
        frame = self.draw_letter_added(

            frame,

            letter_added,

            last_added_letter
        )

        # ------------------------------------------
        # Decorative Separators
        # ------------------------------------------

        frame = self.draw_separators(

            frame

        )

        return frame