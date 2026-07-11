"""
SignLanguageAI
Version 0.7.0

Gesture Detector V2
"""

import math


class GestureDetector:

    def __init__(self):

        self.THUMB_UP = "SPACE"
        self.THUMB_DOWN = "BACKSPACE"
        self.VULCAN = "CLEAR"

    # =====================================================
    # Distance
    # =====================================================

    @staticmethod
    def distance(p1, p2):

        return math.sqrt(

            (p1.x - p2.x) ** 2 +

            (p1.y - p2.y) ** 2

        )

    # =====================================================
    # Finger Extended
    # =====================================================

    @staticmethod
    def finger_extended(

        tip,

        pip,

        mcp

    ):

        return (

            tip.y < pip.y < mcp.y

        )

    # =====================================================
    # Thumb Extended
    # =====================================================

    def thumb_extended(

        self,

        landmarks

    ):

        thumb_tip = landmarks[4]

        thumb_ip = landmarks[3]

        thumb_mcp = landmarks[2]

        wrist = landmarks[0]

        thumb_length = self.distance(

            thumb_tip,

            wrist

        )

        palm_length = self.distance(

            thumb_mcp,

            wrist

        )

        return thumb_length > palm_length * 1.15

    # =====================================================
    # Finger States
    # =====================================================

    def get_finger_states(

        self,

        landmarks

    ):

        states = {

            "thumb":

                self.thumb_extended(

                    landmarks

                ),

            "index":

                self.finger_extended(

                    landmarks[8],

                    landmarks[6],

                    landmarks[5]

                ),

            "middle":

                self.finger_extended(

                    landmarks[12],

                    landmarks[10],

                    landmarks[9]

                ),

            "ring":

                self.finger_extended(

                    landmarks[16],

                    landmarks[14],

                    landmarks[13]

                ),

            "pinky":

                self.finger_extended(

                    landmarks[20],

                    landmarks[18],

                    landmarks[17]

                )

        }

        return states

    # =====================================================
    # Thumb Direction
    # =====================================================

    def thumb_direction(

        self,

        landmarks

    ):

        thumb_tip = landmarks[4]

        thumb_ip = landmarks[3]

        wrist = landmarks[0]

        if thumb_tip.y < wrist.y:

            return "UP"

        if thumb_tip.y > wrist.y:

            return "DOWN"

        if thumb_tip.x > thumb_ip.x:

            return "RIGHT"

        return "LEFT"
        # =====================================================
    # Thumbs Up
    # =====================================================

    def is_thumbs_up(

        self,

        landmarks

    ):

        states = self.get_finger_states(landmarks)

        if not states["thumb"]:
            return False

        if (
            states["index"] or
            states["middle"] or
            states["ring"] or
            states["pinky"]
        ):
            return False

        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]

        return (

            thumb_tip.y < thumb_mcp.y and
            thumb_tip.y < wrist.y

        )
        # =====================================================
    # Thumbs Down
    # =====================================================

    def is_thumbs_down(

        self,

        landmarks

    ):

        states = self.get_finger_states(landmarks)

        if not states["thumb"]:
            return False

        if (
            states["index"] or
            states["middle"] or
            states["ring"] or
            states["pinky"]
        ):
            return False

        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]

        return (

            thumb_tip.y > thumb_mcp.y and
            thumb_tip.y > wrist.y

        )
        # =====================================================
    # Vulcan Salute
    # =====================================================

    def is_vulcan(

        self,

        landmarks

    ):

        states = self.get_finger_states(landmarks)

        if not (

            states["index"] and
            states["middle"] and
            states["ring"] and
            states["pinky"]

        ):

            return False

        gap = self.distance(

            landmarks[12],
            landmarks[16]

        )

        index_gap = self.distance(

            landmarks[8],
            landmarks[12]

        )

        return gap > index_gap * 1.6
    # =====================================================
# Detect Gesture
# =====================================================

def detect(

    self,

    landmarks,

    handedness=None

):

    if landmarks is None:

        return None

    if self.is_thumbs_up(

        landmarks

    ):

        return self.THUMB_UP

    if self.is_thumbs_down(

        landmarks

    ):

        return self.THUMB_DOWN

    if self.is_vulcan(

        landmarks

    ):

        return self.VULCAN

    return None