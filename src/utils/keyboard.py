"""
SignLanguageAI
Version 0.4.1

Professional Keyboard Manager
"""


import cv2


class Keyboard:

    def __init__(self):

        self.previous_key = None

    def get_event(self):

        key = cv2.waitKey(1) & 0xFF

        # No key
        if key == 255:

            self.previous_key = None

            return None

        try:

            key = chr(key).lower()

        except ValueError:

            return None

        # Prevent Windows auto-repeat
        if key == self.previous_key:

            return None

        self.previous_key = key

        # Quit
        if key == "q":

            return {
                "type": "QUIT"
            }

        # Letter
        if "a" <= key <= "z":

            return {
                "type": "LETTER",
                "key": key.upper()
            }

        return {
            "type": "UNKNOWN"
        }