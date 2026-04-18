#!/usr/bin/env python3
"""
Quick YOLOv8 Traffic Sign Model Training Script
Trains directly on LISA or custom traffic sign dataset
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATASET_PATH = 'data/traffic_signs/ts/ts'  # Path to your dataset
MODEL_SIZE = 'yolov8m'  # nano(n), small(s), medium(m), large(l), xlarge(x)
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 0  # GPU device ID

# US Traffic Sign Classes
US_TRAFFIC_SIGNS = {
    'Stop Sign', 'Do Not Enter', 'Yield', 'One Way', 
    'Speed Limit', 'No Left Turn', 'No Right Turn', 'No U-Turn',
    'Pedestrian Crossing', 'School Zone', 'Curve Ahead', 'Slippery Road',
    'Construction', 'Merge', 'Hill Ahead', 'Sharp Turn',
    'Divided Highway', 'Road Narrows', 'Railroad', 'Deer Crossing',
    'Exit', 'Parking', 'Hospital', 'Gas Station'
}

def check_dataset(dataset_path):
    """Verify dataset exists and has correct structure"""
    if not os.path.exists(dataset_path):
        logger.error(f"❌ Dataset not found at {dataset_path}")
        logger.info("Please organize your traffic sign images as:")
        logger.info("  data/traffic_signs/images/train/")
        logger.info("  data/traffic_signs/images/val/")
        logger.info("  data/traffic_signs/labels/train/")
        logger.info("  data/traffic_signs/labels/val/")
        return False
    
    required_dirs = [
        'images/train', 'images/val',
        'labels/train', 'labels/val'
    ]
    
    for dir_name in required_dirs:
        full_path = os.path.join(dataset_path, dir_name)
        if not os.path.exists(full_path):
            logger.error(f"❌ Missing directory: {full_path}")
            return False
    
    logger.info("✅ Dataset structure verified")
    return True

def create_dataset_yaml(dataset_path, num_classes=38):
    """Create dataset.yaml for YOLOv8 training"""
    yaml_content = f"""path: {os.path.abspath(dataset_path)}
train: images/train
val: images/val
nc: {num_classes}

names:
  0: 'Stop Sign'
  1: 'Do Not Enter'
  2: 'Yield'
  3: 'One Way'
  4: 'Speed Limit 5'
  5: 'Speed Limit 15'
  6: 'Speed Limit 25'
  7: 'Speed Limit 35'
  8: 'Speed Limit 45'
  9: 'Speed Limit 55'
  10: 'Speed Limit 65'
  11: 'Speed Limit 75'
  12: 'No Left Turn'
  13: 'No Right Turn'
  14: 'No U-Turn'
  15: 'No Parking'
  16: 'Pedestrian Crossing'
  17: 'School Zone'
  18: 'Curve Ahead'
  19: 'Slippery Road'
  20: 'Construction'
  21: 'Merge'
  22: 'Hill Ahead'
  23: 'Sharp Turn'
  24: 'Divided Highway'
  25: 'Road Narrows'
  26: 'Railroad Crossing'
  27: 'Deer Crossing'
  28: 'Exit'
  29: 'Parking'
  30: 'Hospital'
  31: 'Gas Station'
  32: 'Rest Area'
  33: 'Lodging'
  34: 'Food'
  35: 'Information'
  36: 'Scenic Overlook'
  37: 'Truck Stop'
"""
    
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    logger.info(f"✅ Created dataset.yaml at {yaml_path}")
    return yaml_path

def train_traffic_sign_model(dataset_path):
    """Train YOLOv8 model on traffic signs"""
    
    logger.info("🚀 Starting YOLOv8 Traffic Sign Training")
    logger.info(f"   Model: {MODEL_SIZE}")
    logger.info(f"   Dataset: {dataset_path}")
    logger.info(f"   Epochs: {EPOCHS}")
    logger.info(f"   Batch Size: {BATCH_SIZE}")
    
    # Verify dataset
    if not check_dataset(dataset_path):
        logger.error("❌ Dataset verification failed")
        return None
    
    # Create dataset YAML
    yaml_path = create_dataset_yaml(dataset_path)
    
    try:
        # Load base model
        logger.info(f"📦 Loading {MODEL_SIZE} base model...")
        model = YOLO(f'{MODEL_SIZE}.pt')
        
        # Start training
        logger.info("🔥 Beginning training...")
        results = model.train(
            data=yaml_path,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            patience=20,  # Early stopping
            save=True,
            cache=True,  # Cache images for faster training
            project='traffic_signs_models',
            name=f'usdot_{MODEL_SIZE}_v1',
            verbose=True
        )
        
        logger.info("✅ Training completed!")
        logger.info(f"📊 Results: {results}")
        
        # Export model
        best_model_path = f'traffic_signs_models/usdot_{MODEL_SIZE}_v1/weights/best.pt'
        logger.info(f"💾 Best model saved to: {best_model_path}")
        
        return best_model_path
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        return None

def validate_model(model_path):
    """Validate trained model"""
    try:
        logger.info(f"🔍 Validating model: {model_path}")
        model = YOLO(model_path)
        metrics = model.val()
        logger.info(f"✅ Validation metrics: {metrics}")
        return True
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

def test_inference(model_path, test_image):
    """Test inference on a sample image"""
    try:
        logger.info(f"🧪 Testing inference on: {test_image}")
        model = YOLO(model_path)
        results = model.predict(source=test_image, conf=0.5)
        logger.info(f"✅ Inference successful: {results}")
        return True
    except Exception as e:
        logger.error(f"❌ Inference test failed: {e}")
        return False

if __name__ == '__main__':
    logger.info("╔════════════════════════════════════════════════════════╗")
    logger.info("║   YOLOv8 US Traffic Sign Detection - Training Script   ║")
    logger.info("╚════════════════════════════════════════════════════════╝")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        logger.error(f"❌ Dataset not found at: {DATASET_PATH}")
        logger.info("\n📂 Expected structure:")
        logger.info("   data/traffic_signs/")
        logger.info("   ├── images/")
        logger.info("   │   ├── train/  (your traffic sign images)")
        logger.info("   │   └── val/    (validation images)")
        logger.info("   └── labels/")
        logger.info("       ├── train/  (YOLO format annotations)")
        logger.info("       └── val/")
        logger.info("\n📥 Download datasets from:")
        logger.info("   - LISA: http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html")
        logger.info("   - Kaggle: https://kaggle.com/search?q=traffic+sign")
        sys.exit(1)
    
    # Train model
    best_model = train_traffic_sign_model(DATASET_PATH)
    
    if best_model and os.path.exists(best_model):
        logger.info("\n🎉 Training complete! Next steps:")
        logger.info(f"1. Copy model: cp {best_model} ./usdot_traffic_signs.pt")
        logger.info("2. Update server.py line 28:")
        logger.info("   model = YOLO('usdot_traffic_signs.pt')")
        logger.info("3. Run server: python server.py")
    else:
        logger.error("❌ Training failed - check errors above")
        sys.exit(1)
