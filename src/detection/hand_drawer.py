import cv2

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

FINGERTIPS = [4, 8, 12, 16, 20]


class HandDrawer:

    def draw(self, frame, hand_landmarks):

        h, w, _ = frame.shape

        points = []

        for landmark in hand_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append((x, y))

        # Draw Skeleton
        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 0, 0),
                2
            )

        # Draw Landmarks
        for i, point in enumerate(points):

            color = (0,255,0)

            if i in FINGERTIPS:
                color = (0,0,255)

            cv2.circle(
                frame,
                point,
                6,
                color,
                -1
            )

        # Bounding Box
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        cv2.rectangle(
            frame,
            (min(xs)-20, min(ys)-20),
            (max(xs)+20, max(ys)+20),
            (255,255,0),
            2
        )

        return frame