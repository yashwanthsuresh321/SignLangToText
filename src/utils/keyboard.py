"""
SignLanguageAI
Version 0.7.0

Professional Keyboard Manager
"""


import cv2


class Keyboard:

    def __init__(self):

        self.previous_key = None

    def get_event(self):

        key = cv2.waitKey(1) & 0xFF
        # -------------------------
        # ESC = Quit
        # -------------------------

        if key == 27:

            self.previous_key = None

            return {

                "type": "QUIT"

            }

    # -------------------------
    # No key
    # -------------------------

        if key == 255:
 
            self.previous_key = None

            return None

        try:

            key = chr(key).lower()

        except ValueError:

            return None

    # -------------------------
    # Prevent Windows auto-repeat
    # -------------------------

        if key == self.previous_key:

            return None

        self.previous_key = key

   

    # -------------------------
    # Letters
    # -------------------------

        if "a" <= key <= "z" and key not in ("j", "z"):

            return {

                "type": "LETTER",

                "key": key.upper()

            }

    # -------------------------
    # Command Labels
    # -------------------------

        if key == "1":

            return {

                "type": "COMMAND",

                "key": "SPACE"

            }

        if key == "2":

            return {

                "type": "COMMAND",

                "key": "BACKSPACE"

            }

        if key == "3":

            return {

                "type": "COMMAND",

                "key": "CLEAR"

            }


        # -------------------------
        # Dynamic Recording
        # -------------------------

        if key == "j":
            return {
                "type": "DYNAMIC_RECORD",
                "label": "J"
            }

        if key == "z":
            return {
                "type": "DYNAMIC_RECORD",
                "label": "Z"
            }

        if key == "s":
            return {
                "type": "SAVE_DYNAMIC_DATASET"
            }

        return {

            "type": "UNKNOWN"

        }