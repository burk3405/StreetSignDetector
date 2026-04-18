# US Traffic Sign Detection - Training Guide

This guide explains how to train or use a custom YOLOv8 model specifically for US Department of Transportation (USDOT) signs.

## Quick Option: Use Pre-trained Traffic Sign Model

If you want to skip training, download a pre-trained traffic sign model:

### **Option A: TensorFlow/Keras Traffic Sign Model**

```bash
# Download pre-trained model trained on USDOT signs
cd c:\Users\acbur\OneDrive\Desktop\Livestream
curl -L "https://github.com/username/traffic-sign-detector/releases/download/v1.0/usdot-yolov8-pretrained.pt" -o usdot-yolov8.pt
```

Then update `server.py` line 28:
```python
model = YOLO('usdot-yolov8.pt')  # Your trained model
```

---

## Full Option: Train Custom Model on Traffic Signs

### **Step 1: Get a Traffic Sign Dataset**

Use one of these publicly available datasets:

#### **LISA Traffic Sign Dataset** (Recommended)
- Comprehensive US traffic signs
- 47 sign types
- ~6,235 images
- Download: http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html

```bash
# After downloading and extracting LISA dataset
# Convert to YOLOv8 format (instructions below)
```

#### **USDOT Traffic Sign Dataset**
- Official US traffic signs
- Specifically trained for autonomous vehicles
- Download various sources:
  - Kaggle: https://www.kaggle.com/datasets/muhammadbilalahmed/traffic-signs-data
  - GitHub: Traffic Sign Recognition datasets

#### **German Traffic Sign Dataset** (Also applicable)
- 43 classes (similar to US signs)
- Good for transfer learning
- Download: http://benchmark.ini.rub.de/

### **Step 2: Prepare Dataset in YOLOv8 Format**

YOLOv8 requires images and annotations in this structure:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── val/
│       ├── img100.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt  (YOLO format: <class_id> <x> <y> <w> <h>)
    │   └── ...
    └── val/
        └── ...
```

**YOLO label format example (`img1.txt`):**
```
11 0.5 0.5 0.3 0.4
5 0.7 0.2 0.15 0.25
```
(class_id, center_x_norm, center_y_norm, width_norm, height_norm)

### **Step 3: Create Training Script**

Create `train_traffic_signs.py`:

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8m.pt')  # medium model for balance

# Train on traffic signs dataset
results = model.train(
    data='path/to/dataset.yaml',  # See Step 4
    epochs=100,
    imgsz=640,
    device=0,  # GPU device ID (0 for first GPU)
    batch=16,
    patience=20,  # Early stopping
    save=True,
    project='traffic_signs',
    name='usdot_detection_v1'
)

# Export best model
best_model = YOLO('traffic_signs/usdot_detection_v1/weights/best.pt')
best_model.export(format='pt')
```

### **Step 4: Create Dataset YAML**

Create `dataset.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 38  # Number of traffic sign classes

names:
  0: 'Stop Sign'
  1: 'Do Not Enter'
  2: 'Yield'
  3: 'One Way'
  4: 'Speed Limit 5'
  5: 'Speed Limit 15'
  # ... add all 38 classes
```

### **Step 5: Train the Model**

```bash
cd c:\Users\acbur\OneDrive\Desktop\Livestream

# Run training
python train_traffic_signs.py
```

This will:
- Download YOLOv8 base weights
- Train on your traffic sign dataset
- Save best model to `traffic_signs/usdot_detection_v1/weights/best.pt`
- Show training progress, mAP scores, etc.

### **Step 6: Use Trained Model**

Update `server.py` line 28 with your trained model:

```python
def load_traffic_sign_model():
    try:
        # Use your trained traffic sign model!
        model = YOLO('traffic_signs/usdot_detection_v1/weights/best.pt')
        logger.info('✅ Traffic Sign Model loaded')
        return model
    except Exception as e:
        logger.error(f'❌ Failed to load: {e}')
        return None
```

---

## Expected Detection Results

After training, your model should detect:

### **Regulatory Signs** (Red/White) - RED BOXES
- 🛑 Stop Sign
- ⛔ Do Not Enter  
- 🚦 Yield
- ➡️ One Way
- 🔄 Turn Prohibition Signs
- 🚫 No Parking
- ⏱️ Speed Limit Signs (5-75 mph)

### **Warning Signs** (Yellow/Black) - YELLOW BOXES
- 🚶 Pedestrian Crossing
- 🏫 School Zone
- 🔄 Curve Ahead
- ❄️ Slippery Road
- 🚧 Construction
- 🦌 Deer Crossing
- ⛰️ Hill/Grade

### **Informational Signs** (Green/White) - GREEN BOXES
- Exit
- Parking
- Hospital
- Gas Station
- Rest Area

---

## Training Performance Benchmarks

Expected metrics after ~100 epochs:

| Metric | Value |
|--------|-------|
| **mAP50** | 0.85-0.92 |
| **mAP50-95** | 0.72-0.85 |
| **Precision** | 0.88-0.95 |
| **Recall** | 0.80-0.90 |
| **FPS** | 30-60 (GPU) |

---

## Deployment: Use Trained Model

Once trained, your model file is production-ready:

```bash
# Copy to project
cp traffic_signs/usdot_detection_v1/weights/best.pt ./

# Run server with trained model
python server.py
```

The frontend will automatically use your trained model!

---

## Advanced: Fine-tuning from Pre-trained

If using transfer learning (recommended):

```python
# Start from traffic sign pre-trained model
model = YOLO('path/to/pretrained_traffic_signs.pt')

# Fine-tune on your specific data
results = model.train(
    data='dataset.yaml',
    epochs=50,  # Fewer epochs needed
    device=0,
    batch=32,  # Can use larger batch
)
```

---

## Troubleshooting

### **Low accuracy (< 0.75 mAP)**
- Add more training data (300+ images per class)
- Increase image resolution: `imgsz=1280`
- Use larger model: `yolov8l.pt` or `yolov8x.pt`
- Increase epochs: `epochs=150+`

### **Training too slow**
- Use GPU: `device=0`
- Smaller batch size: `batch=8`
- Skip validation: `val=False` (not recommended)
- Use nano model for testing: `yolov8n.pt`

### **Model not detecting signs**
- Dataset format incorrect - verify YOLO label format
- Wrong class count in `dataset.yaml`
- Insufficient training data
- Classes not balanced

### **CUDA/GPU errors**
- Install CUDA: https://developer.nvidia.com/cuda-downloads
- Install cuDNN
- Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

---

## References

- **Ultralytics YOLOv8 Docs:** https://docs.ultralytics.com/
- **LISA Dataset Origin:** http://cvrr.ucsd.edu/LISA/
- **YOLO Training Guide:** https://docs.ultralytics.com/modes/train/
- **OpenCV for Sign Processing:** https://docs.opencv.org/

---

## Next Steps

1. ⬇️ Download a traffic sign dataset
2. 📂 Organize in YOLOv8 format
3. ▶️ Run training script
4. ✅ Use trained model in `server.py`
5. 🚀 Deploy to production!

Good luck training! 🚗🛑
