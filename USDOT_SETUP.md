# 🛑 US Traffic Sign Detection System (USDOT)

Real-time detection and recognition of US Department of Transportation (USDOT) traffic signs for autonomous vehicles.

## ✨ Features

- **🛑 Regulatory Signs:** Stop, Do Not Enter, Yield, One Way, Speed Limits
- **⚠️ Warning Signs:** Pedestrian, School Zone, Curves, Slippery Road
- **🟢 Informational:** Exit, Parking, Hospital, Gas Station  
- **📊 Real-time Display:** Color-coded boxes (Red=Stop, Yellow=Warning, Green=Info)
- **📈 FPS Counter:** Monitor detection performance
- **🌓 Dark/Light Mode:** Eye-friendly interface

---

## 🚀 Quick Start (3 Steps)

### **Step 1: Install Dependencies**

```bash
cd c:\Users\acbur\OneDrive\Desktop\Livestream
pip install -r requirements.txt
```

### **Step 2: Start the Backend**

```bash
python server.py
```

You'll see:
```
✅ YOLOv8 US Traffic Sign Detection Server on http://localhost:5000
🛑 Detecting: Stop, Do Not Enter, Speed Limits, One Way, and more...
```

### **Step 3: Open Frontend**

```bash
python -m http.server 8000
```

Then open: **http://localhost:8000**

**That's it!** Start recording your webcam and watch it detect traffic signs.

---

## 🎯 Setup for USDOT-Specific Detection

Currently the backend uses a generic YOLOv8 model. To get **dedicated USDOT sign detection**, choose one:

### **Option A: Use Pre-trained USDOT Model** (Easiest)

Download a pre-trained traffic sign model and copy to project:

```bash
# Download pre-trained traffic sign model (you can find these on GitHub/Kaggle)
# Place the .pt file in the project directory
cp path/to/traffic_signs_model.pt ./usdot_traffic_signs.pt
```

Then update `server.py` (line ~28):

```python
def load_traffic_sign_model():
    try:
        model = YOLO('usdot_traffic_signs.pt')  # ← Your model!
        logger.info('✅ USDOT Traffic Sign Model loaded')
        return model
    except Exception as e:
        logger.error(f'❌ Failed to load: {e}')
        return None
```

Restart the server and you're done!

---

### **Option B: Train Your Own Model** (Best for Accuracy)

**Step 1: Get a Traffic Sign Dataset**

Choose one:

- **LISA Traffic Sign Dataset** (Best for US roads)
  - Download: http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
  - 47 sign types, 6,235 images
  - Exactly for US roads

- **Kaggle Traffic Signs**
  - https://www.kaggle.com/datasets/muhammadbilalahmed/traffic-signs-data
  - Multiple datasets available

- **German Traffic Sign Dataset** (Transfer learning)
  - http://benchmark.ini.rub.de/
  - 43 classes, highly accurate

**Step 2: Organize Dataset in YOLOv8 Format**

Your downloaded dataset needs this structure:

```
data/traffic_signs/
├── images/
│   ├── train/
│   │   ├── stop_1.jpg
│   │   ├── stop_2.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── labels/
    ├── train/
    │   ├── stop_1.txt  (YOLO format)
    │   └── ...
    └── val/
        └── ...
```

**YOLO label format** (`stop_1.txt`):
```
0 0.5 0.5 0.3 0.4
```
(class_id, center_x_norm, center_y_norm, width_norm, height_norm)

**Step 3: Run Training**

```bash
python train_traffic_signs.py
```

The script will:
- ✅ Validate dataset format
- 📦 Load YOLOv8 base model  
- 🔥 Train on your traffic sign data
- 💾 Save best model to `traffic_signs_models/`

**Step 4: Use Your Trained Model**

```bash
# Copy the trained model
cp traffic_signs_models/usdot_yolov8m_v1/weights/best.pt ./usdot_traffic_signs.pt

# Update server.py line 28
# model = YOLO('usdot_traffic_signs.pt')

# Restart server
python server.py
```

---

## 📊 Detected Signs

### **Regulatory Signs** (Red bounding box)
- ✋ Stop Sign
- ⛔ Do Not Enter
- 🙏 Yield
- ➡️ One Way (left, right, up)
- 🚫 No Left Turn
- 🚫 No Right Turn  
- 🚫 No U-Turn
- 🚫 No Parking
- 🚗 Speed Limit (5, 15, 25, 35, 45, 55, 65, 75 mph)

### **Warning Signs** (Yellow bounding box)
- 🚶 Pedestrian Crossing
- 🏫 School Zone
- 🔄 Curve Ahead
- ❄️ Slippery Road
- 🔗 Merge
- ⛰️ Hill Ahead
- 🔄 Sharp Turn
- 🚧 Construction Ahead
- 🦌 Deer Crossing
- 🚂 Railroad Crossing

### **Informational Signs** (Green bounding box)
- 🅿️ Parking
- 🚪 Exit
- 🏥 Hospital
- ⛽ Gas Station
- 🛏️ Lodging
- 🍽️ Food
- ℹ️ Information
- 🚙 Rest Area

---

## ⚙️ Configuration

### **Change Confidence Threshold**

Use the slider in the UI (controls minimum detection confidence)

### **Switch Model Size**

Edit `server.py` line ~28:

```python
# Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
model = YOLO('yolov8m.pt')    # Medium (default, balanced)
model = YOLO('yolov8s.pt')    # Small (faster)
model = YOLO('yolov8l.pt')    # Large (more accurate)
```

### **Enable GPU Acceleration**

```python
# In server.py, add device=0:
results = model(image, conf=confidence_threshold, device=0, verbose=False)
```

---

## 🧪 Testing Your Setup

### **Test Backend**

```bash
# In a new terminal
curl http://localhost:5000/api/health
```

Should return:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "YOLOv8 (Traffic Signs)",
  "supported_signs": 38
}
```

### **Test Supported Signs**

```bash
curl http://localhost:5000/api/supported-signs
```

### **Test Video Feed**

1. Open http://localhost:8000
2. Allow camera access
3. Point at traffic signs
4. Green bounding boxes = Detection!

---

## 🐛 Troubleshooting

### **"Failed to connect to backend"**
- Is `server.py` running? Run it in a terminal
- Is frontend pointing to `http://localhost:5000`?
- Check firewall isn't blocking port 5000

### **"Model not detecting signs"**
- Using generic YOLOv8? It won't detect specific signs well
- **Solution:** Train or download a traffic-sign-specific model
- Increase confidence slider slowly to see if detections exist at low confidence

### **Train script fails**
- Dataset format wrong? Check YOLO format
- Out of GPU memory? Reduce `BATCH_SIZE` in script
- Missing dataset? Check `data/traffic_signs/` exists

### **Slow inference (< 10 FPS)**
- Use smaller model: `yolov8s.pt` instead of `yolov8m.pt`
- Reduce image resolution: `imgsz=416` in training
- Enable GPU: `device=0`

---

## 📈 Performance Expectations

| Model | Speed (FPS) | Accuracy (mAP50) | Memory |
|-------|-------------|-----------------|--------|
| **YOLOv8n** (nano) | 40-60 | ~0.75 | 200MB |
| **YOLOv8s** (small) | 30-40 | ~0.82 | 300MB |
| **YOLOv8m** (medium) | 15-30 | ~0.88 | 500MB |
| **YOLOv8l** (large) | 8-15 | ~0.92 | 1GB |
| **Trained USDOT** | 20-40 | ~0.90+ | 500MB |

---

## 📚 File Guide

| File | Purpose |
|------|---------|
| `server.py` | Flask backend (YOLOv8 inference) |
| `app.js` | Frontend detection logic |
| `index.html` | Web UI |
| `styles.css` | Styling |
| `train_traffic_signs.py` | Training script for custom models |
| `TRAINING_GUIDE.md` | Detailed training instructions |
| `requirements.txt` | Python dependencies |

---

## 🎓 Learning Resources

- **Ultralytics YOLOv8:** https://docs.ultralytics.com/
- **LISA Traffic Signs:** http://cvrr.ucsd.edu/LISA/
- **YOLO Training:** https://docs.ultralytics.com/modes/train/
- **Traffic Sign Recognition Paper:** https://arxiv.org/
- **Real-world Autonomous Driving:** https://github.com/carla-simulator/carla

---

## 📝 Next Steps

1. ✅ Run the system with generic YOLOv8 (what you have now)
2. 📥 Download LISA or Kaggle traffic sign dataset
3. 🔥 Train model with `train_traffic_signs.py`
4. 🚀 Deploy trained model for production accuracy
5. 🌐 Host on cloud for remote autonomous vehicle inference

---

## 💡 Pro Tips

- **For testing:** Use YOLOv8 small or medium models (fast enough)
- **For production:** Train on specific dataset (much better accuracy)
- **For embedded:** Convert to TensorFlow Lite or ONNX for edge devices
- **For cloud:** Deploy to AWS/GCP with GPU instances

---

## 📞 Support

If you have issues:
1. Check error messages in terminal
2. Ensure all dependencies installed: `pip install -r requirements.txt`
3. Verify dataset format for training
4. Check that both server and frontend are running on correct ports

---

**Happy detecting! 🛑🚦✨**
