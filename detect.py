"""
Real-Time Tomato Detection using YOLO

This script loads a trained YOLO model and performs real-time inference
on webcam frames. Detected tomatoes and keypoints are displayed live.

Usage:
    python detect.py

    or

    python detect.py --model best.pt --conf 0.5 --camera 0

Controls:
    Press 'q' to quit.
"""

from pathlib import Path
import argparse
import logging

import cv2
from ultralytics import YOLO


WINDOW_NAME = "Tomato Detection"


def setup_logging() -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )


def load_model(model_path: str) -> YOLO:
    """
    Load the trained YOLO model.

    Args:
        model_path: Path to the trained model (.pt)

    Returns:
        Loaded YOLO model.

    Raises:
        FileNotFoundError: If model file does not exist.
        RuntimeError: If model cannot be loaded.
    """
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        model = YOLO(str(path))
        logging.info(f"Loaded model: {path}")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model.\n{e}") from e


def run_detection(model: YOLO, confidence: float, camera: int) -> None:
    """
    Run real-time webcam detection.

    Args:
        model: Loaded YOLO model.
        confidence: Detection confidence threshold.
        camera: Camera index.
    """
    cap = cv2.VideoCapture(camera)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    logging.info("Webcam started.")
    logging.info("Press 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                logging.warning("Failed to read frame.")
                break

            # Run inference
            results = model(frame, conf=confidence)

            # Draw detections
            annotated_frame = results[0].plot()

            # Display
            cv2.imshow(WINDOW_NAME, annotated_frame)

            # Exit when q is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Resources released.")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Tomato Detection using YOLO"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="Path to the trained YOLO model (.pt)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold (default: 0.4)"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)"
    )

    return parser.parse_args()


def main() -> None:
    """Main function."""
    setup_logging()

    args = parse_arguments()

    try:
        model = load_model(args.model)
        run_detection(model, args.conf, args.camera)

    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    main()