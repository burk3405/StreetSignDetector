# Real-Time US Traffic Sign Detection System Using YOLOv8 and OAK-D Lite Camera

---

## Abstract

This report presents the design and implementation of a real-time US Department of Transportation (USDOT) traffic sign detection system built on the YOLOv8 object detection framework with integrated OAK-D Lite stereo depth camera support. The system captures live video, performs inference using a YOLOv8 medium model, classifies detected signs into regulatory, warning, and informational categories, and overlays color-coded bounding boxes with distance estimates onto a browser-based user interface. A Flask backend serves as the bridge between the OAK-D Lite hardware, the YOLOv8 inference engine, and a responsive HTML/JavaScript frontend. The system supports 38 distinct US traffic sign classes, provides 3D spatial coordinate estimation via stereo depth, and includes a virtual camera subsystem that exposes the processed feed as a standard webcam device for use in video conferencing applications. This project demonstrates an end-to-end pipeline for real-time traffic sign recognition suitable for autonomous driving research, driver assistance prototyping, and intelligent transportation system development.

---

## 1. Introduction

Traffic sign detection and recognition is a critical component in autonomous driving systems, advanced driver-assistance systems (ADAS), and intelligent transportation infrastructure. Accurate, real-time identification of regulatory signs (e.g., Stop, Speed Limit), warning signs (e.g., Pedestrian Crossing, Curve Ahead), and informational signs (e.g., Hospital, Exit) directly impacts vehicle safety and navigation decisions.

This project addresses the challenge of building a complete, functional traffic sign detection pipeline that operates in real time. The system combines several modern technologies:

- **YOLOv8 (You Only Look Once, Version 8)** — Ultralytics' state-of-the-art object detection model, providing high-accuracy inference at real-time speeds.
- **OAK-D Lite Camera** — Luxonis' stereo depth camera with an onboard vision processing unit (VPU), enabling both RGB capture and per-pixel depth estimation.
- **Flask Web Server** — A lightweight Python backend that orchestrates camera input, model inference, and API communication.
- **Browser-Based Frontend** — An HTML5/CSS3/JavaScript interface rendering live detection results with bounding boxes, confidence scores, distance readouts, and a detection event log.

The goal of this project is to deliver a modular, extensible system capable of detecting 38 classes of US traffic signs, estimating their physical distance from the camera, and presenting results through an intuitive web interface — all running at interactive frame rates.

---

## 2. Methodology

### 2.1 System Architecture

The system follows a client-server architecture with three primary layers:

1. **Hardware Layer** — OAK-D Lite stereo camera captures synchronized RGB and depth frames at 1280×720 resolution and 30 FPS.
2. **Backend Layer** — A Python Flask server manages camera initialization, runs YOLOv8 inference, performs sign classification heuristics, computes spatial coordinates, and exposes RESTful API endpoints.
3. **Frontend Layer** — A single-page web application consumes the API, renders the MJPEG video feed, overlays detection canvas drawings, and displays a real-time detection log.

### 2.2 Object Detection Model

The project uses **YOLOv8m (medium)** as the base detection model, loaded via the Ultralytics Python library. YOLOv8 is a single-stage anchor-free detector that predicts bounding boxes and class probabilities in a single forward pass, making it well-suited for real-time applications.

Key model parameters:
- **Model variant:** YOLOv8m (medium) — balancing speed and accuracy
- **Input resolution:** 640×640 pixels
- **Confidence threshold:** User-adjustable (default 0.50)
- **Post-processing:** Non-Maximum Suppression (NMS) with IoU threshold of 0.45

The system defines 38 US traffic sign classes organized into four categories:

| Category | Examples | Color Code |
|---|---|---|
| **Regulatory** (0–15) | Stop Sign, Do Not Enter, Yield, One Way, Speed Limits (5–75 mph), No Left/Right Turn, No U-Turn, No Parking | Red |
| **Warning** (16–27) | Pedestrian Crossing, School Zone, Curve Ahead, Slippery Road, Merge, Construction, Deer Crossing, Sharp Turn | Yellow |
| **Informational** (28–33) | Exit, Parking, Hospital, Gas Station, Lodging, Rest Area | Green |
| **Guide/Direction** (34–37) | Route Shield, Interstate Shield, US Route Shield, State Route Shield | Blue |

### 2.3 Sign Classification Heuristics

In addition to the neural network's class predictions, the system applies a secondary HSV color-space analysis on each detected region of interest (ROI) to refine classification. This heuristic examines:

- **Red pixel ratio** — High red content indicates regulatory signs (Stop, Do Not Enter, Yield). Aspect ratio further differentiates between them.
- **Yellow pixel ratio** — Dominant yellow indicates warning signs.
- **White pixel ratio** — High white content suggests directional signs (One Way).

This dual-classification approach (neural network + color heuristic) improves robustness when using a general-purpose model prior to fine-tuning on a dedicated traffic sign dataset.

### 2.4 Depth Estimation and Spatial Coordinates

The OAK-D Lite camera provides synchronized stereo depth maps aligned to the RGB frame. For each detected sign, the system:

1. Extracts the bounding box center point.
2. Queries the depth map at that location to obtain distance in millimeters.
3. Computes 3D spatial coordinates (X, Y, Z) using approximate camera intrinsics (70° horizontal FOV, 55° vertical FOV).
4. Reports distance in both meters and feet.

Depth statistics (min, max, mean, median) are also computed across the entire bounding box ROI to provide robust distance estimates even when individual depth pixels are noisy.

### 2.5 Training Pipeline

A dedicated training script (`train_traffic_signs.py`) is provided for fine-tuning YOLOv8 on custom traffic sign datasets. The training pipeline supports:

- **Datasets:** LISA Traffic Sign Dataset (47 classes, ~6,235 images), Kaggle USDOT datasets, German Traffic Sign Benchmark (for transfer learning)
- **Data format:** YOLOv8-compatible directory structure with normalized bounding box annotations (`<class_id> <cx> <cy> <w> <h>`)
- **Hyperparameters:** 100 epochs, batch size 16, image size 640, early stopping with patience of 20 epochs
- **Output:** Trained `.pt` model weights saved to `traffic_signs_models/` directory

### 2.6 Frontend Design

The web interface provides:

- **Live Video Feed** — MJPEG stream from the OAK-D camera rendered in an `<img>` element.
- **Detection Overlay** — HTML5 Canvas layer drawing color-coded bounding boxes with class labels, confidence percentages, and distance annotations.
- **Confidence Slider** — Real-time adjustment of the detection threshold (0.00–1.00).
- **FPS Counter** — Live frames-per-second measurement.
- **Detection Log Console** — Scrollable event log with timestamped entries, sign category styling, and recommended driver actions (e.g., "STOP: Apply brakes immediately").
- **Dark/Light Mode** — Toggleable interface theme with persistent preference storage.
- **Virtual Camera Controls** — Start/stop buttons for the virtual webcam feature.

### 2.7 Virtual Camera Subsystem

The system includes a virtual camera module that rebroadcasts the OAK-D feed as a standard webcam device using the `pyvirtualcam` library with OBS Virtual Camera as the backend driver. This allows the processed camera feed (with detection overlays) to appear as a selectable camera in applications such as Google Meet, Zoom, and Microsoft Teams.

---

## 3. System Pipeline

The following describes the step-by-step data flow from camera capture to user display:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM PIPELINE                              │
└─────────────────────────────────────────────────────────────────────┘

  1. CAPTURE          OAK-D Lite camera captures RGB + Depth frames
        │              (1280×720 @ 30 FPS)
        ▼
  2. INGEST           oak_d_integration.py reads frames via DepthAI SDK
        │              into a thread-safe queue
        ▼
  3. ADAPT            oak_d_adapter.py converts frames to base64 JPEG,
        │              caches depth data for spatial queries
        ▼
  4. SERVE            server.py (Flask) exposes:
        │                • /video_feed    → MJPEG live stream
        │                • /api/oak-d/detect → YOLOv8 inference endpoint
        │                • /api/oak-d/status → Camera status
        ▼
  5. DETECT           YOLOv8m model runs inference on each frame
        │                • Outputs bounding boxes + class IDs + confidence
        ▼
  6. CLASSIFY         HSV color heuristic refines sign type
        │                • Red → Regulatory  |  Yellow → Warning
        │                • White → Directional  |  Green → Info
        ▼
  7. LOCALIZE         Depth map queried at each detection's bbox
        │                • Computes X, Y, Z spatial coordinates
        │                • Reports distance in meters and feet
        ▼
  8. RESPOND          Flask returns JSON with detections, spatial data,
        │              and classification to the frontend
        ▼
  9. RENDER           app.js draws color-coded bounding boxes on
        │              HTML5 Canvas overlay atop the MJPEG video feed
        ▼
  10. LOG             Detection console logs timestamped events with
                       action recommendations (e.g., "STOP", "YIELD")
```

---

## 4. Results

The system successfully achieves:

- **Real-Time Performance:** The detection loop operates at interactive frame rates, with the Flask backend processing frames and returning results within the browser's `requestAnimationFrame` cycle.
- **38-Class Support:** The system defines and can classify 38 distinct US traffic sign types spanning regulatory, warning, informational, and guide categories.
- **3D Spatial Awareness:** When the OAK-D Lite camera is connected, the system provides per-detection distance estimates and 3D coordinates, enabling spatial reasoning about sign positions relative to the vehicle.
- **Depth Statistics:** Bounding-box-level depth statistics (min, max, mean, median) provide robust distance measurements that are resilient to noisy depth pixels.
- **Color-Coded Visualization:** The frontend uses intuitive color coding — red for regulatory signs, yellow for warnings, and green for informational — matching real-world USDOT sign color conventions.
- **Action Recommendations:** The detection log generates context-aware driving recommendations (e.g., "Apply brakes immediately" for Stop signs, "Adjust speed to limit" for Speed Limit signs).
- **Virtual Camera Integration:** The OAK-D feed can be broadcast as a system webcam, enabling real-time traffic sign detection feeds to be shared in video conferencing tools.
- **Dual-Mode Operation:** The system supports both OAK-D Lite hardware and a browser-only ONNX Runtime fallback, ensuring broad accessibility.

### Key Technical Metrics

| Metric | Value |
|---|---|
| Supported sign classes | 38 |
| RGB resolution | 1280 × 720 |
| Frame rate target | 30 FPS |
| Model size (YOLOv8m) | ~42 MB |
| Default confidence threshold | 0.50 |
| NMS IoU threshold | 0.45 |
| Depth range (OAK-D Lite) | 0.2 m – 19.1 m |
| Spatial FOV (H × V) | 70° × 55° |

---

## 5. OAK-D Lite Camera Implementation

### 5.1 Hardware Overview

The OAK-D Lite is a compact stereo depth camera manufactured by Luxonis, featuring:

- A 4K RGB center camera for color image capture
- Dual mono cameras for stereo depth computation
- An Intel Movidius Myriad X Vision Processing Unit (VPU) for onboard neural inference

### 5.2 Software Integration

The camera integration is implemented across three Python modules:

1. **`oak_d_integration.py`** — Low-level camera driver using the DepthAI SDK (`depthai` v2.24.1). This module:
   - Connects to the OAK-D device
   - Configures RGB and depth pipelines
   - Runs a background thread that continuously captures frames into a thread-safe queue (max size 2 to prevent latency buildup)
   - Exposes methods for point-level and region-level depth queries

2. **`oak_d_adapter.py`** — Adapter layer that bridges the OAK-D camera to the Flask server. Responsibilities include:
   - Converting raw RGB frames to base64-encoded JPEG for API transmission
   - Caching frame data for subsequent spatial queries within the same request cycle
   - Computing 3D spatial coordinates (X, Y, Z) from 2D bounding boxes using depth data and approximate camera intrinsics
   - Managing the virtual camera lifecycle (start/stop)

3. **`virtual_camera.py`** — Virtual webcam driver using `pyvirtualcam` (v0.4.1). This module:
   - Creates a virtual camera device at the configured resolution and frame rate
   - Runs a dedicated streaming thread that pulls frames from the OAK-D adapter and pushes them to the virtual device
   - Handles frame format conversion (grayscale → BGR, BGRA → BGR, resizing)
   - Requires OBS Virtual Camera as the backend driver on Windows

### 5.3 Graceful Degradation

The system is designed to operate with or without the OAK-D camera:

- If the OAK-D is unavailable or fails to initialize, the system logs a warning and falls back to API-based webcam capture.
- The `USE_OAK_D` environment variable allows disabling OAK-D support entirely.
- Import errors for `depthai` or `pyvirtualcam` are caught gracefully, and dependent features are disabled without crashing the server.

---

## 6. Conclusion

This project demonstrates a complete, functional pipeline for real-time US traffic sign detection, combining deep learning–based object detection (YOLOv8), stereo depth estimation (OAK-D Lite), and an interactive web-based visualization layer. The modular architecture — separating camera hardware abstraction, inference logic, and frontend rendering — allows individual components to be upgraded independently. The inclusion of a training pipeline enables fine-tuning on specialized traffic sign datasets for improved accuracy beyond the base COCO-pretrained model. The virtual camera subsystem extends the system's utility beyond standalone use, enabling integration into video conferencing and remote monitoring workflows.

Future work could include:
- Fine-tuning on the LISA Traffic Sign Dataset for dedicated USDOT sign recognition accuracy
- Deploying the YOLOv8 model directly on the OAK-D's Myriad X VPU for edge inference
- Integrating GPS data for location-aware sign context (e.g., expected speed limits by road)
- Adding temporal tracking (e.g., SORT or DeepSORT) to maintain sign identities across frames
- Expanding to additional sign standards (EU, Asian road sign conventions)

---

## 7. Acknowledgements

This project was built with the following open-source technologies and communities:

- **Ultralytics** — For the YOLOv8 object detection framework and pre-trained model weights
- **Luxonis** — For the OAK-D Lite camera hardware and the DepthAI SDK
- **OpenCV** — For image processing, color space conversion, and frame encoding
- **Flask** — For the lightweight Python web server framework
- **NumPy** — For numerical computation and array operations
- **pyvirtualcam** — For virtual camera device creation
- **OBS Studio** — For the virtual camera backend driver on Windows
- **LISA Traffic Sign Dataset (UCSD)** — For publicly available US traffic sign training data
- **COCO Dataset** — For the pre-trained model weights used as a baseline

---

## Appendix A: Technology Stack

| Component | Technology | Version |
|---|---|---|
| Object Detection | Ultralytics YOLOv8 | 8.0.226 |
| Camera Hardware | OAK-D Lite (Luxonis) | — |
| Camera SDK | DepthAI | 2.24.1 |
| Image Processing | OpenCV | 4.8.1.78 |
| Web Server | Flask | 3.0.0 |
| CORS Middleware | Flask-CORS | 4.0.0 |
| Numerical Computing | NumPy | 1.24.3 |
| Virtual Camera | pyvirtualcam | 0.4.1 |
| Frontend | HTML5, CSS3, JavaScript (ES6+) | — |

## Appendix B: API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend HTML page |
| `/video_feed` | GET | MJPEG live stream from OAK-D camera |
| `/api/detect` | POST | Run YOLOv8 detection on a base64-encoded image |
| `/api/health` | GET | Backend health check and model status |
| `/api/oak-d/frame` | GET | Get current RGB frame as base64 JPEG |
| `/api/oak-d/detect` | GET | Run detection on current OAK-D frame with spatial data |
| `/api/oak-d/status` | GET | Camera initialization and capability status |
| `/api/supported-signs` | GET | List all 38 supported traffic sign classes |
| `/api/virtual-camera/start` | POST | Activate virtual webcam device |
| `/api/virtual-camera/stop` | POST | Deactivate virtual webcam device |
| `/api/virtual-camera/status` | GET | Virtual camera running state |

## Appendix C: File Structure

| File | Purpose |
|---|---|
| `server.py` | Flask backend — routes, inference, camera management |
| `app.js` | Frontend application — detection loop, canvas rendering, UI |
| `index.html` | Frontend HTML structure |
| `styles.css` | Frontend styling and theme |
| `oak_d_integration.py` | Low-level OAK-D camera driver (DepthAI) |
| `oak_d_adapter.py` | OAK-D ↔ Flask adapter with spatial computation |
| `virtual_camera.py` | Virtual webcam device management |
| `training.py` | Basic YOLOv8 training script (COCO8) |
| `train_traffic_signs.py` | Traffic sign–specific training pipeline |
| `requirements.txt` | Python dependency list |
| `yolov8m.pt` | YOLOv8 medium pre-trained weights |
| `yolov8n.pt` | YOLOv8 nano pre-trained weights |
| `data/traffic_signs/` | Training data directory (labels, configs) |
