import cv2

# ==========================================================
# MediaPipe Hand Connections
# ==========================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FINGERTIPS = [4, 8, 12, 16, 20]


# ==========================================================
# Hand Drawer
# ==========================================================

class HandDrawer:
    """
    SignLanguageAI
    Version 1.0.0

    Draws normalized MediaPipe landmarks directly onto the
    display frame.

    MediaPipe landmarks are normalized (0–1), so no scaling
    from processing resolution is required.
    """

    def draw(self, frame, hand_landmarks):
        """
        Draw hand landmarks on the given display frame.

        Parameters
        ----------
        frame : np.ndarray
            Display frame.

        hand_landmarks : list
            MediaPipe normalized landmarks.

        Returns
        -------
        np.ndarray
            Frame with landmarks drawn.
        """

        if hand_landmarks is None:
            return frame

        h, w = frame.shape[:2]

        points = []

        # --------------------------------------------------
        # Convert normalized landmarks to display pixels
        # --------------------------------------------------

        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))

        # --------------------------------------------------
        # Draw Connections
        # --------------------------------------------------

        for start, end in HAND_CONNECTIONS:
            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

        # --------------------------------------------------
        # Draw Landmarks
        # --------------------------------------------------

        for index, point in enumerate(points):

            color = (0, 255, 0)

            if index in FINGERTIPS:
                color = (0, 0, 255)

            cv2.circle(
                frame,
                point,
                6,
                color,
                -1,
                cv2.LINE_AA
            )

        # --------------------------------------------------
        # Draw Bounding Box
        # --------------------------------------------------

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        padding = 20

        cv2.rectangle(
            frame,
            (min(xs) - padding, min(ys) - padding),
            (max(xs) + padding, max(ys) + padding),
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        return frame