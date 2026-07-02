# src/camera.py

import cv2


def open_camera(index: int):
    cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    return cap


def release_camera(cap):
    cap.release()
    cv2.destroyAllWindows()