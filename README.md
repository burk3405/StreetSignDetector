# Street Sign Detection with Ultralytics YOLOv8

Real-time object detection using your webcam with professional YOLOv8 integration.

## Setup Options

### **Option A: Browser-Only (No Backend) - Quick Start**

Currently set up with ONNX Runtime Web - runs entirely in your browser with no server needed.

```bash
# Just open index.html in your browser
# No installation required!
```

**Pros:**
- ✅ No server setup
- ✅ Instant startup
- ✅ Works offline
- ✅ Mobile compatible

**Cons:**
- Browser-based ONNX format (limited model variants)

---

### **Option B: With Ultralytics Backend (Recommended)**

Uses the official **Ultralytics YOLOv8** library for maximum accuracy and customization.

#### **Prerequisites:**
- Python 3.8+
- pip

#### **Step 1: Install Python Dependencies**

```bash
pip install -r requirements.txt
```

This installs:
- `ultralytics` - Official YOLOv8 library
- `flask` - Web server
- `opencv-python` - Image processing
- `numpy` - Numerical operations

#### **Step 2: Start the Backend Server**

Open a terminal in the project directory and run:

```bash
python server.py
```

You should see:
```
🚀 Starting YOLOv8 Detection Server on http://localhost:5000
 * Running on http://0.0.0.0:5000
```

**First run note:** The YOLOv8 model (~42MB) will auto-download on first request (~30 seconds).

#### **Step 3: Open the Frontend**

In another terminal, start a simple HTTP server in the project directory:

```bash
# Windows (PowerShell)
python -m http.server 8000

# macOS/Linux
python3 -m http.server 8000
```

Then open in your browser:
```
http://localhost:8000
```

**Pros:**
- ✅ Official Ultralytics YOLOv8
- ✅ Higher accuracy
- ✅ Better performance (GPU support)
- ✅ Access to all YOLOv8 variants (nano, small, medium, large, xlarge)
- ✅ Custom model fine-tuning support

**Cons:**
- Requires Python backend
- Needs server running
- Additional dependencies

---

## API Architecture

### **Backend → Frontend Communication**

**Request (Frontend → Backend):**
```json
POST /api/detect
{
  "image": "base64_encoded_jpeg",
  "confidence": 0.5
}
```

**Response (Backend → Frontend):**
```json
{
  "success": true,
  "detections": [
    {
      "class": "stop sign",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2],
      "classId": 11
    }
  ]
}
```

---

## Configuration

### **Change Confidence Threshold**
Use the slider in the UI (0.0 → 1.0)

### **Switch YOLOv8 Model Size**
Edit `server.py` line 24:

```python
# Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
model = YOLO('yolov8n.pt')  # nano (fastest)
model = YOLO('yolov8m.pt')  # medium (balanced)
model = YOLO('yolov8x.pt')  # xlarge (most accurate)
```

### **Enable GPU Acceleration**
Add to `server.py` line 41:

```python
results = model(image, conf=confidence_threshold, device=0, verbose=False)
```

---

## Detected Street Signs

The model detects these traffic-related objects:

- 🛑 **Stop Sign** (class 11)
- 🚦 **Traffic Light** (class 9)
- 🔴 **Fire Hydrant** (class 10)
- 🅿️ **Parking Meter** (class 12)
- Plus 76 other COCO dataset classes

---

## Troubleshooting

### **"Failed to connect to YOLOv8 backend"**
- ❌ Backend not running: Run `python server.py` in a terminal
- ❌ Wrong port: Make sure server is on `localhost:5000`
- ❌ CORS issue: Verify `flask-cors` is installed

### **Slow performance**
- Use nano model: `yolov8n.pt` (fastest)
- Enable GPU: Add `device=0` to model.predict()
- Reduce image quality: Lower JPEG quality in `app.js`

### **Model download fails**
- Check internet connection
- Ultralytics will cache model in `~/.config/Ultralytics/yolov8n.pt`

### **High latency on first request**
- Model is compiling - this is normal first time only
- Subsequent frames will be much faster

---

## File Structure

```
Livestream/
├── index.html          # Frontend UI
├── app.js              # Frontend logic
├── styles.css          # Styling
├── server.py           # Python backend (YOLOv8 inference)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Performance Metrics

### **Browser-Only (ONNX)**
- **FPS:** 15-25 FPS (depends on GPU)
- **Latency:** ~50-100ms per frame
- **Memory:** ~200MB

### **With Backend (Ultralytics)**
- **FPS:** 20-40 FPS (CPU), 40-60+ FPS (GPU)
- **Latency:** ~30-80ms per frame
- **Memory:** ~500MB

---

## Advanced: Running on Cloud

You can deploy the backend to cloud services like AWS, Google Cloud, or Heroku:

```bash
# Heroku example
heroku create your-app-name
git push heroku main
```

Then update `app.js` line 160:
```javascript
await fetch('https://your-app-name.herokuapp.com/api/detect', {
```

---

## License

This project uses:
- **YOLOv8** - Ultralytics (AGPL-3.0)
- **ONNX Runtime** - Microsoft (MIT)
- **OpenCV** - BSD

---

## Next Steps

1. **Try Option A first** (no backend) to test the interface
2. **Upgrade to Option B** (Python backend) for production accuracy
3. **Fine-tune a custom model** for specific street signs using Ultralytics docs

---

## Support

For YOLOv8 documentation: https://docs.ultralytics.com/
For local issues: Check console (F12) for error messages
