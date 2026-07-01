# 🍅 Real-Time Tomato Detection and Keypoint Estimation using YOLO

A computer vision project that performs real-time tomato detection and keypoint estimation using a custom-trained YOLO model and a webcam. The application captures live video, runs inference on each frame, and displays annotated detections with bounding boxes and keypoints.

---

## Features

* Real-time webcam inference
* Custom YOLO model for tomato detection
* Keypoint estimation
* Adjustable confidence threshold
* Clean and modular Python implementation
* Easy-to-use command-line interface


## Project Structure

```
Tomato-Detection/
│
├── detect.py              # Main application
├── best.pt                # Trained YOLO model
├── requirements.txt       # Python dependencies
├── README.md



---

## Installation

### 1. Clone the repository: https://github.com/imankhah1993-code/tomato-keypoint-detection.git


### 2. Create a virtual environment (recommended)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application using the default settings:

```bash
python detect.py
```

Specify a different model:

```bash
python detect.py --model best.pt
```

Set a different confidence threshold:

```bash
python detect.py --conf 0.5
```

Use another camera:

```bash
python detect.py --camera 1
```

Example:

```bash
python detect.py --model best.pt --conf 0.45 --camera 0
```

---

## Controls

| Key   | Action               |
| ----- | -------------------- |
| **Q** | Quit the application |

---

## Model

The project uses a **custom-trained YOLO pose model** capable of:

* Detecting tomatoes
* Predicting tomato keypoints
* Performing inference in real time

The trained weights are loaded from:

```
best.pt
```

---

## Technologies Used

* Python 3
* Ultralytics YOLO
* OpenCV
* PyTorch
* NumPy

---

## Future Improvements

* Save annotated videos
* Image inference support
* Video file inference
* FPS counter
* Object tracking
* Performance benchmarking
* Export predictions to CSV or JSON
* GUI for selecting models and camera sources

---


## Acknowledgements

* Ultralytics YOLO
* OpenCV
* PyTorch

---

## Author

Iman Mirzakhah

GitHub: https://github.com/imankhah1993-code
