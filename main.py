# main.py

import cv2
from src.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    CAMERA_INDEX,
    WINDOW_NAME,
)

from src.camera import open_camera, release_camera
from src.detector import TomatoDetector


def run():
    detector = TomatoDetector(MODEL_PATH)
    cap = open_camera(CAMERA_INDEX)

    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame")
                break

            annotated_frame = detector.predict(frame, CONFIDENCE_THRESHOLD)

            cv2.imshow(WINDOW_NAME, annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        release_camera(cap)


if __name__ == "__main__":
    run()